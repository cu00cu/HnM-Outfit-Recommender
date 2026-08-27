"""
Outfit Recommender System - H&M real-data version
===================================================
Adapted from Practical 8 (Parts I, II, III) - movie recommender -> outfit recommender.

The structure below follows the practicals directly:
  Section 1 : Data Preparation            (Practical 8 Part II, Section 1)
  Section 2 : Features Generation         (Practical 8 Part II, Section 2)
  Section 3 : Content-Based Recommender   (Practical 8 Part II, Section 3)
  Section 4 : Collaborative Filtering     (Practical 8 Part III)
  Section 5 : Hybrid (content + collaborative)

Main change from the movie version: a movie recommender returns items in the
SAME category (Spider-Man -> more superhero films). An outfit recommender must
return a DIFFERENT category (a shirt -> trousers/skirts), so we filter the
candidates to complementary categories before ranking them.

This version uses REAL data throughout, including for collaborative filtering:
  sample_data/styles.csv       - built from H&M's real articles.csv
  sample_data/images/<id>.jpg  - real H&M product photos, resized
  sample_data/interactions.csv - REAL customer purchase records (customer_id,
                                  article_id pairs), filtered from H&M's real
                                  31.8-million-row transaction log. This is
                                  what the earlier version had to simulate,
                                  because the previous dataset (Kaggle Fashion
                                  Product Images) had no linked purchase data.
Exported from the Kaggle "H&M Personalized Fashion Recommendations" dataset.
"""

import os
import colorsys
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix

st.set_page_config(page_title="Outfit Recommender", layout="wide")

DATA_DIR = "sample_data"
CSV_PATH = os.path.join(DATA_DIR, "styles.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")
INTERACTIONS_PATH = os.path.join(DATA_DIR, "interactions.csv")

# ---------------------------------------------------------------------------
# Clothing category rules (the outfit-specific part of this project)
#
# H&M's real product_type_name has many more distinct values than the old
# dataset (dozens of upper-body garment names alone), so rather than listing
# every one, we use the "slot" the export script already assigned from H&M's
# product_group_name (Garment Upper body / Lower body / Full body / Shoes) -
# a small, stable set of 4 categories.
# ---------------------------------------------------------------------------
# Which slots COMPLETE a given slot
COMPLEMENT_SLOTS = {
    "top":    ["bottom", "shoe"],
    "bottom": ["top", "shoe"],
    "dress":  ["shoe"],
    "shoe":   ["top", "bottom", "dress"],
}

# Which genders may be mixed. A women's top may be paired with women's or
# unisex items, but never with menswear.
#
# H&M's children's-wear ("Baby/Children") was excluded during export, since
# adult outfit-matching logic doesn't apply cleanly to kids' sizing, so only
# three groups remain here.
GENDER_GROUPS = {
    "Men":    ["Men", "Unisex"],
    "Women":  ["Women", "Unisex"],
    "Unisex": ["Unisex", "Men", "Women"],
}

# What a sensible outfit looks like: if the user gives us a top, they want
# mostly bottoms plus a couple of shoes - not five pairs of shoes.
OUTFIT_QUOTA = {
    "top": {"bottom": 3, "shoe": 2},
    "bottom": {"top": 3, "shoe": 2},
    "dress": {"shoe": 5},
    "shoe": {"top": 2, "bottom": 2, "dress": 1},
}

# ---------------------------------------------------------------------------
# "Look" (dress style) - DERIVED, not a raw column in the dataset.
# ---------------------------------------------------------------------------
# Style ("look") is precomputed by the export script from H&M's real
# section_name / garment_group_name columns, so no derivation is needed here.
# derive_look() is kept only as a fallback for older/manually-edited CSVs
# that might not already have a "look" column.
def derive_look(row):
    text = f"{row.get('usage','')}".lower()
    if "sport" in text:
        return "Sportswear / Gym"
    if "loung" in text or "home" in text:
        return "Loungewear / Home"
    if "formal" in text or "office" in text:
        return "Office / Formal"
    return "Everyday / Streetwear"


# Colours that go with anything
NEUTRALS = {
    "Black", "White", "Off White", "Grey", "Grey Melange", "Charcoal",
    "Navy Blue", "Beige", "Brown", "Coffee Brown", "Cream", "Silver",
    "Khaki", "Taupe", "Tan", "Nude", "Steel", "Mushroom Brown",
    "Metal", "Mole",
}

# Position of each colour on the colour wheel (like hours on a clock).
# Used to tell analogous colours (neighbours) from clashing ones.
COLOUR_WHEEL = {
    "Red": 0, "Maroon": 0, "Burgundy": 0, "Rust": 1, "Orange": 1, "Peach": 1,
    "Copper": 1, "Bronze": 1, "Gold": 2, "Yellow": 2, "Mustard": 2,
    "Lime Green": 3, "Fluorescent Green": 3, "Olive": 3, "Khaki Green": 3,
    "Green": 4, "Sea Green": 5, "Teal": 6, "Turquoise Blue": 6, "Turquoise": 6,
    "Blue": 7, "Navy Blue": 7, "Lavender": 9, "Purple": 9, "Mauve": 9,
    "Magenta": 10, "Pink": 11, "Rose": 11,
}


def season_ok(s1, s2):
    """Seasons match if they are the same, or in the same warm/cold pair.

    Used as an ELIGIBILITY gate in find_candidates() - unchanged, since
    that filter is already tested. season_score() below is a separate,
    graded version used only in RANKING, so an exact season match can
    outscore a merely-adjacent one instead of both counting equally.
    """
    return s1 == s2 or {"Fall", "Winter"} <= {s1, s2} or {"Spring", "Summer"} <= {s1, s2}


def season_score(s1, s2):
    """Graded season compatibility for ranking: 1.0 exact, 0.6 adjacent
    (e.g. Spring/Summer), 0.0 opposite (e.g. Summer/Winter)."""
    if s1 == s2:
        return 1.0
    if {"Fall", "Winter"} <= {s1, s2} or {"Spring", "Summer"} <= {s1, s2}:
        return 0.6
    return 0.0


# How much each signal contributes to the content-based ranking score.
# Kept as named constants (rather than inline numbers) so they can be
# grid-searched against real evaluation data - see tune_weights.py.
CONTENT_WEIGHT_STYLE = 0.4
CONTENT_WEIGHT_COLOUR = 0.3
CONTENT_WEIGHT_SEASON = 0.3


def _classify_colour(name):
    """Resolve a real colour name to either ('neutral', None) or
    ('wheel', position), tolerating case differences and compound names.

    Exact matches (case-insensitive) are always checked first, across BOTH
    lists, before any substring fallback - this matters because a compound
    name can otherwise be wrongly captured by a shorter, unrelated substring.
    For example 'Khaki green' contains the substring 'khaki', which is
    itself a listed neutral colour, so a naive substring check classified
    it as a neutral instead of resolving its own, more specific 'Khaki
    Green' entry in the hue wheel. Checking exact matches first, and
    preferring the LONGEST substring match when falling back, avoids this.
    """
    nl = name.lower()
    if any(n.lower() == nl for n in NEUTRALS):
        return ("neutral", None)
    for k, v in COLOUR_WHEEL.items():
        if k.lower() == nl:
            return ("wheel", v)

    best = None  # (match_length, kind, value)
    for n in NEUTRALS:
        if n.lower() in nl and (best is None or len(n) > best[0]):
            best = (len(n), "neutral", None)
    for k, v in COLOUR_WHEEL.items():
        if k.lower() in nl and (best is None or len(k) > best[0]):
            best = (len(k), "wheel", v)
    if best:
        return (best[1], best[2])
    return (None, None)


def _is_neutral(name):
    kind, _ = _classify_colour(name)
    return kind == "neutral"


def _wheel_position(name):
    kind, value = _classify_colour(name)
    return value if kind == "wheel" else None


def colour_score(c1, c2):
    """How well two colours work together, from 0.2 (clash) to 0.95 (classic).

    The earlier version returned 1.0 whenever either colour was neutral, which
    made every candidate score the same, so colour stopped affecting the
    ranking at all. This version gives a graded score instead.
    """
    n1, n2 = _is_neutral(c1), _is_neutral(c2)

    if n1 and n2:
        return 0.95 if c1 != c2 else 0.85      # e.g. white + navy: classic
    if n1 or n2:
        return 0.85                            # a neutral with a colour: safe
    if c1 == c2:
        return 0.60                            # all one colour can look flat

    p1, p2 = _wheel_position(c1), _wheel_position(c2)
    if p1 is None or p2 is None:
        return 0.50                            # unknown colour: neither good nor bad

    gap = min(abs(p1 - p2), 12 - abs(p1 - p2))  # distance around the wheel
    if gap <= 1:
        return 0.78                            # neighbours: harmonious
    if gap == 2:
        return 0.65
    if gap >= 5:
        return 0.72                            # opposites: deliberate contrast
    return 0.30                                # awkward middle distance: clash


# ---------------------------------------------------------------------------
# Section 1 - Data Preparation      (Practical 8 Part II, Section 1)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load styles.csv, keep only items we have rules and images for."""
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    df = df[df["slot"].isin(COMPLEMENT_SLOTS.keys())]
    df = df.dropna(subset=["gender", "articleType", "baseColour", "season", "slot"])
    df = df[df["id"].apply(lambda i: os.path.exists(os.path.join(IMG_DIR, f"{i}.jpg")))]
    df = df.reset_index(drop=True)
    if "look" not in df.columns:
        df["look"] = df.apply(derive_look, axis=1)   # fallback only
    return df


# ---------------------------------------------------------------------------
# Section 2 - Features Generation   (Practical 8 Part II, Section 2)
# The movie version used the 'overview' text. Here we build the same kind of
# text from the item's attributes, then TF-IDF + cosine similarity.
# ---------------------------------------------------------------------------
@st.cache_data
def build_similarity(df):
    soup = (
        df["gender"] + " " + df["usage"] + " " + df["season"] + " "
        + df["baseColour"].str.replace(" ", "") + " " + df["articleType"]
        + " " + df["look"].str.replace(r"[ /]", "", regex=True)
        + " " + df["productDisplayName"].fillna("")
    )
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(soup)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)   # same as practical
    return cosine_sim


# ---------------------------------------------------------------------------
# Section 3 - Content-Based Recommender   (Practical 8 Part II, Section 3)
# Same steps as the practical: get index -> score all items -> sort -> top K.
# Added step: keep only COMPLEMENTARY categories, and add a colour score.
# ---------------------------------------------------------------------------
def find_candidates(df, item, prefs):
    """Candidate items = complementary slots only.

    The filters come from what the USER selected (prefs), not from whichever
    catalogue item we matched them to. Gender is never dropped: relaxing it
    is what previously caused menswear to be suggested for a women's top.

    Look is relaxed PER COMPLEMENTARY SLOT, not on the combined pool. A top
    query needs both bottoms and shoes; if 200 bottoms match the look but
    only 1 shoe does, checking the combined total (201) never triggers
    relaxation, so the shoe slot stays starved even though shoes are
    genuinely scarce for that look. Checking each slot on its own catches
    this: bottoms stay strictly look-matched (there are plenty), shoes fall
    back to the wider gendered pool (because there weren't enough).
    """
    base = df[df["slot"].isin(COMPLEMENT_SLOTS[item["slot"]])]

    allowed_genders = GENDER_GROUPS.get(prefs["gender"], [prefs["gender"], "Unisex"])
    gendered = base[base["gender"].isin(allowed_genders)]
    if gendered.empty:
        gendered = base          # only if the sample has nothing for this gender

    MIN_PER_SLOT = 3
    parts = []
    for slot in COMPLEMENT_SLOTS[item["slot"]]:
        slot_pool = gendered[gendered["slot"] == slot]
        slot_looked = slot_pool[slot_pool["look"] == prefs["look"]] if prefs.get("look") else slot_pool
        if len(slot_looked) < MIN_PER_SLOT:
            slot_looked = slot_pool      # relax ONLY this slot, only because IT is scarce
        parts.append(slot_looked)
    looked = pd.concat(parts) if parts else gendered

    strict = looked[looked["season"].apply(lambda s: season_ok(s, prefs["season"]))]
    if len(strict) >= 5:
        return strict.copy()

    return looked.copy()


def get_recommendations(item_id, df, cosine_sim, indices, prefs, k=5, offset=0):
    idx = indices[item_id]
    item = df.iloc[idx]

    candidates = find_candidates(df, item, prefs)
    if candidates.empty:
        return []

    positions = indices[candidates["id"]].values
    style = cosine_sim[idx, positions]
    # colour and season are compared against what the USER chose, not the matched item
    colour = candidates["baseColour"].apply(lambda c: colour_score(prefs["colour"], c)).values
    season = candidates["season"].apply(lambda s: season_score(s, prefs["season"])).values

    candidates["score"] = (
        CONTENT_WEIGHT_STYLE * style
        + CONTENT_WEIGHT_COLOUR * colour
        + CONTENT_WEIGHT_SEASON * season
    )
    return balanced_top_k(candidates, item["slot"], k, offset)


def balanced_top_k(candidates, query_slot, k=5, offset=0):
    """Build a sensible outfit instead of just taking the top K scores.

    For a top we want roughly 3 bottoms and 2 shoes. Taking the raw top K
    lets one group (usually shoes) fill every slot.
    """
    candidates = candidates.sort_values("score", ascending=False)

    quota = OUTFIT_QUOTA.get(query_slot, {})
    chosen = []
    for slot, count in quota.items():
        in_slot = candidates[candidates["slot"] == slot]["id"].tolist()
        if not in_slot:
            continue
        # `offset` lets the user ask for another outfit: we step further down
        # the ranked list instead of returning the same items again.
        # Never ask for more items than this slot actually has, or the
        # wrap-around below returns the SAME item more than once (measured:
        # 100 of 1440 swept combinations returned a duplicate product before
        # this guard, e.g. a scarce Loungewear bottoms pool).
        take = min(count, len(in_slot))
        start = (offset * take) % len(in_slot)
        picks = (in_slot + in_slot)[start:start + take]
        chosen.extend(picks)

    # if a slot had too few items, top up with the next best of anything
    for item_id in candidates["id"]:
        if len(chosen) >= k:
            break
        if item_id not in chosen:
            chosen.append(item_id)

    return chosen[:k]


# ---------------------------------------------------------------------------
# Section 4 - Collaborative Filtering   (Practical 8 Part III)
# The practical read real MovieLens user ratings via pivot_table + corrwith().
# This version reads REAL H&M customer purchase records (interactions.csv,
# exported from the 31.8-million-row transactions_train.csv, filtered to our
# sampled catalogue) - genuine collaborative filtering on real behaviour, not
# a simulation.
# ---------------------------------------------------------------------------
@st.cache_data
def build_collaborative(df):
    """Build an item-item similarity matrix from real H&M purchase records.

    Same idea as the practical's pivot_table, but built as a SPARSE matrix.
    A dense users x items table would allocate a cell for every possible
    customer/product combination - with hundreds of thousands of real
    customers that runs to tens of gigabytes, and almost every cell is a
    zero because a shopper only ever buys a handful of items. The sparse
    version stores only the purchases that actually happened.
    """
    if not os.path.exists(INTERACTIONS_PATH):
        return pd.DataFrame()

    interactions = pd.read_csv(INTERACTIONS_PATH)
    interactions = interactions[interactions["item_id"].isin(df["id"])]
    if interactions.empty:
        return pd.DataFrame()

    interactions = interactions.drop_duplicates(subset=["user_id", "item_id"])

    # map ids onto row/column positions for the sparse matrix
    users = interactions["user_id"].astype("category")
    items = interactions["item_id"].astype("category")

    user_item = csr_matrix(
        (np.ones(len(interactions), dtype=np.float32),
         (users.cat.codes.to_numpy(), items.cat.codes.to_numpy())),
        shape=(len(users.cat.categories), len(items.cat.categories)),
    )

    # The practical used corrwith(), which returns NaN on sparse data.
    # cosine_similarity on the same matrix is the same idea without the NaN.
    item_sim = cosine_similarity(user_item.T, dense_output=True)
    item_ids = list(items.cat.categories)
    return pd.DataFrame(item_sim, index=item_ids, columns=item_ids)


def collaborative_recommendations(item_id, df, item_sim, prefs, k=5, offset=0):
    if item_sim.empty or item_id not in item_sim.index:
        return []

    scores = item_sim[item_id].drop(index=item_id, errors="ignore")
    scores = scores[scores > 0]
    if scores.empty:
        return []

    item = df[df["id"] == item_id].iloc[0]
    allowed_genders = GENDER_GROUPS.get(prefs["gender"], [prefs["gender"], "Unisex"])

    candidates = df[
        df["id"].isin(scores.index)
        & df["slot"].isin(COMPLEMENT_SLOTS[item["slot"]])
        & df["gender"].isin(allowed_genders)
    ].copy()

    # Keep the outfit in one style, same as the other two methods - but
    # per complementary slot, same reasoning as find_candidates(): a
    # combined check can leave a scarce slot (e.g. shoes) with nothing.
    if prefs.get("look"):
        MIN_PER_SLOT = 3
        parts = []
        for slot in COMPLEMENT_SLOTS[item["slot"]]:
            slot_pool = candidates[candidates["slot"] == slot]
            slot_styled = slot_pool[slot_pool["look"] == prefs["look"]]
            if len(slot_styled) < MIN_PER_SLOT:
                slot_styled = slot_pool
            parts.append(slot_styled)
        if parts:
            candidates = pd.concat(parts)

    if candidates.empty:
        return []

    candidates["score"] = candidates["id"].map(scores)
    return balanced_top_k(candidates, item["slot"], k, offset)


# ---------------------------------------------------------------------------
# Section 5 - Hybrid: average the content score and the collaborative score
# ---------------------------------------------------------------------------
def hybrid_recommendations(item_id, df, cosine_sim, indices, item_sim, prefs, k=5, offset=0):
    idx = indices[item_id]
    item = df.iloc[idx]

    candidates = find_candidates(df, item, prefs)
    if candidates.empty:
        return []

    positions = indices[candidates["id"]].values
    style = cosine_sim[idx, positions]
    colour = candidates["baseColour"].apply(lambda c: colour_score(prefs["colour"], c)).values
    season = candidates["season"].apply(lambda s: season_score(s, prefs["season"])).values
    candidates["content"] = (
        CONTENT_WEIGHT_STYLE * style
        + CONTENT_WEIGHT_COLOUR * colour
        + CONTENT_WEIGHT_SEASON * season
    )

    if not item_sim.empty and item_id in item_sim.index:
        candidates["collab"] = candidates["id"].map(item_sim[item_id]).fillna(0)
    else:
        candidates["collab"] = 0.0

    for col in ["content", "collab"]:
        spread = candidates[col].max() - candidates[col].min()
        candidates[col] = (candidates[col] - candidates[col].min()) / spread if spread > 0 else 0

    candidates["score"] = 0.5 * candidates["content"] + 0.5 * candidates["collab"]
    return balanced_top_k(candidates, item["slot"], k, offset)


# ---------------------------------------------------------------------------
# Helper: detect the main colour of an uploaded photo
# ---------------------------------------------------------------------------
# Hue anchors around the colour wheel, used to name the colour in a photo.
# Comparing raw RGB distance does not work: a pale pink is numerically closer
# to Silver and Beige than to Pink, which is why light garments were being
# misread. Hue survives lightness changes, so we classify on hue instead.
HUE_NAMES = [
    (0, "Red"), (20, "Orange"), (45, "Yellow"), (75, "Lime Green"),
    (120, "Green"), (160, "Sea Green"), (180, "Teal"), (215, "Blue"),
    (270, "Purple"), (300, "Magenta"), (330, "Pink"), (355, "Red"),
]


def classify_pixel(rgb):
    """Name a single pixel's colour using hue, saturation and brightness."""
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360

    # HSV saturation is a ratio, so it becomes unreliable as brightness drops:
    # when the brightest channel is already tiny, a difference of only a few
    # RGB points produces a high saturation reading. A real photo of black
    # leggings with a faint cool cast measured saturation 0.38 - well past the
    # 0.12 neutral threshold below - and was therefore named "Purple". The
    # rule here is that the darker a pixel is, the more saturated it must be
    # before its hue is believed at all.
    if v < 0.12:
        return "Black"                             # too dark to read any hue
    if v < 0.35 and s < 0.47:
        # 0.47 sits between the highest saturation measured on real black
        # fabric with a colour cast (0.40) and the lowest measured on a
        # genuinely dark colour such as navy (0.545).
        return "Black" if v < 0.25 else "Charcoal"  # dark with only a weak cast

    if s < 0.12:                                   # barely any colour: a neutral
        # Thresholds widened from an earlier version tuned only against pure
        # synthetic colour swatches. Real phone photos of black fabric rarely
        # measure near-zero brightness - camera auto-exposure lifts shadows
        # to preserve visible detail, and fabric folds catch ambient light -
        # so a narrow "v < 0.18" threshold was misreading real black garments
        # as Charcoal or Grey. Verified against realistic photo brightness
        # levels before widening.
        if v < 0.35:
            return "Black"
        if v < 0.55:
            return "Charcoal"
        if v < 0.75:
            return "Grey"
        if v < 0.90:
            return "Silver"
        return "White"

    name = min(HUE_NAMES, key=lambda x: min(abs(hue - x[0]), 360 - abs(hue - x[0])))[1]

    if name == "Red":
        if v < 0.45:
            return "Maroon"
        if s < 0.35 and v > 0.75:
            return "Pink"                          # a pale tint of red is pink
    if name == "Orange" and v < 0.5:
        return "Brown"
    if name == "Yellow" and s < 0.3:
        return "Beige"
    if name == "Yellow" and v < 0.6:
        return "Mustard"                           # a dark yellow reads as mustard
    if name == "Lime Green" and (s < 0.50 or v < 0.60):
        # "Lime Green" implies a vivid, bright colour. A muted or darker
        # yellow-green is an olive. Measured on a real photograph of a
        # mustard t-shirt under fluorescent lighting: hue 70 degrees (which
        # genuinely sits in the yellow-green band) but saturation only 0.32
        # and brightness 0.39, so naming it "Lime Green" was misleading even
        # though the hue itself was measured correctly.
        return "Olive"
    if name == "Blue" and v < 0.4:
        return "Navy Blue"
    if name == "Pink" and v < 0.5:
        return "Maroon"
    return name


# Approximate brightness (0 = black, 1 = white) for every neutral, so two
# neutrals can be compared by how similar they actually look rather than
# treated as equally close just because both are "a neutral". Without this,
# a dark colour like Navy Blue or Black matched EVERY neutral at an equal
# distance of zero, so the nearest-match search simply returned whichever
# neutral happened to sort first alphabetically (Beige) - wrong regardless
# of what was actually detected.
NEUTRAL_BRIGHTNESS = {
    "Black": 0.10, "Navy Blue": 0.20, "Charcoal": 0.30, "Coffee Brown": 0.30,
    "Mole": 0.35, "Brown": 0.40, "Mushroom Brown": 0.45, "Steel": 0.50,
    "Grey": 0.55, "Grey Melange": 0.55, "Metal": 0.55, "Khaki": 0.60,
    "Taupe": 0.60, "Tan": 0.65, "Nude": 0.70, "Beige": 0.75,
    "Silver": 0.80, "Cream": 0.85, "Off White": 0.90, "White": 0.95,
}


# The 16 everyday colour names offered to the user. Each maps to a name the
# scoring logic already understands, so the picker can use plain language
# ("Navy") without the user needing to know H&M's own vocabulary
# ("Navy Blue", "Grey Melange", "Mole").
STANDARD_COLOURS = {
    "Black": "Black", "White": "White", "Grey": "Grey", "Silver": "Silver",
    "Beige": "Beige", "Brown": "Brown", "Navy": "Navy Blue", "Maroon": "Maroon",
    "Red": "Red", "Pink": "Pink", "Orange": "Orange", "Yellow": "Yellow",
    "Green": "Green", "Teal": "Teal", "Blue": "Blue", "Purple": "Purple",
}


def _interpretations(name):
    """All valid readings of a colour name.

    Some names legitimately belong to both groups - "Navy Blue" is a
    wearable neutral AND a real position on the blue side of the colour
    wheel. Returning both lets the distance function pick whichever
    interpretation makes the two colours most comparable, instead of
    forcing one reading and wrongly reporting an unrelated-colour gap.
    """
    readings = []
    nl = name.lower()

    is_neutral = any(n.lower() == nl for n in NEUTRALS) or \
                 any(n.lower() in nl for n in NEUTRALS)
    if is_neutral:
        readings.append(("neutral", NEUTRAL_BRIGHTNESS.get(name, 0.5)))

    pos = None
    for k, v in COLOUR_WHEEL.items():
        if k.lower() == nl:
            pos = v
            break
    if pos is None:
        for k, v in COLOUR_WHEEL.items():
            if k.lower() in nl:
                pos = v
                break
    if pos is not None:
        readings.append(("wheel", pos))

    return readings


def colour_distance(a, b):
    """How far apart two colour names are: 0 = identical, 12 = opposite.

    Neutrals are compared by brightness, hues by their position on the
    colour wheel. When a name has more than one valid reading, the closest
    pairing wins.
    """
    ra, rb = _interpretations(a), _interpretations(b)
    if not ra or not rb:
        return 6.0

    best = None
    for ka, va in ra:
        for kb, vb in rb:
            if ka == "neutral" and kb == "neutral":
                d = abs(va - vb) * 12
            elif ka == "wheel" and kb == "wheel":
                d = float(min(abs(va - vb), 12 - abs(va - vb)))
            else:
                d = 6.0
            if best is None or d < best:
                best = d
    return best


def detect_colours(image, secondary_ratio=0.55):
    """Read the garment's colour(s) from the photo.

    Returns (primary, secondary). `secondary` is only set when a second
    colour is nearly as common as the first, which happens with genuinely
    two-tone garments - a half-black, half-white shirt measured 43% and 41%
    across the two, where reporting only the winner would be misleading.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    box = (int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.8))
    pixels = np.array(rgb.crop(box).resize((80, 80))).reshape(-1, 3)

    counts = Counter(classify_pixel(p) for p in pixels)
    ranked = [(name, n) for name, n in counts.most_common()]
    if not ranked:
        return None, None

    # white/silver are usually the backdrop rather than the garment
    garment = [(n, c) for n, c in ranked if n not in ("White", "Silver")] or ranked
    primary, primary_count = garment[0]

    secondary = None
    for name, count in garment[1:]:
        # only flag a genuinely different colour, not a neighbouring shade
        if count >= primary_count * secondary_ratio and colour_distance(primary, name) >= 3:
            secondary = name
            break

    return primary, secondary


def resolve_query_colour(detected, chosen):
    """Combine what the photo shows with what the user picked.

    If the two broadly agree, the detected name wins because it is more
    specific - a photo read as "Navy Blue" is more precise than a user
    picking the general "Blue". If they clearly disagree, the user's choice
    wins, since they can see the real garment and the camera may have been
    misled by lighting or background.
    """
    if not detected:
        return chosen
    return detected if colour_distance(detected, chosen) <= 2.5 else chosen


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Styling. Ink-blue and warm sand: Fraunces (a high-contrast display face) for
# the wordmark, Inter for everything else. Deliberately conservative - it only
# styles our own elements and never reaches into Streamlit's internal DOM,
# because overriding Streamlit's own containers breaks rendering whenever the
# deployed version differs from the local one.
#
# IMPORTANT: no blank lines are allowed inside the <style> block below. In
# Markdown a blank line ends a raw HTML block, so everything after it would be
# printed on the page as plain text instead of being applied as CSS.
# ---------------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
.block-container { padding-top: 3rem; max-width: 1500px; }
.brass-rule { height: 3px; width: 52px; background: #B07D4F; margin-bottom: 16px; }
.wordmark { font-family: 'Fraunces', serif; font-size: 46px; font-weight: 700; color: #12212E; letter-spacing: -1px; line-height: 1.05; margin: 0; }
.tagline { font-family: 'Inter', sans-serif; font-size: 15px; color: #63707E; margin: 8px 0 0 0; max-width: 46ch; }
.count { font-family: 'Inter', sans-serif; font-size: 11px; color: #93A0AC; letter-spacing: 2px; text-transform: uppercase; margin-top: 10px; }
.hairline { height: 1px; background: #E2DACD; margin: 22px 0 6px 0; }
.step { font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #B07D4F; margin: 16px 0 6px 0; }
.placeholder { border: 1px dashed #D8CDBC; border-radius: 6px; padding: 46px 28px; font-family: 'Inter', sans-serif; font-size: 14px; color: #63707E; line-height: 1.65; text-align: center; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

if not os.path.exists(CSV_PATH) or not os.path.isdir(IMG_DIR):
    st.error(
        "Product data not found.\n\n"
        "This app needs the real Kaggle product photos. Add a folder named "
        "`sample_data` to the repository containing `styles.csv` and an "
        "`images` folder, then redeploy."
    )
    st.stop()

df = load_data()
if df.empty:
    st.error("styles.csv loaded, but no rows had a matching image in sample_data/images.")
    st.stop()

@st.cache_data
def image_display_width(ids):
    """Pick a display size that suits the photos we actually have.

    Small source photos are shown smaller so they stay sharp; larger photos are
    shown bigger. This means swapping in the high-resolution dataset improves
    the display automatically, with no code change.
    """
    widths = []
    for i in list(ids)[:25]:
        path = os.path.join(IMG_DIR, f"{i}.jpg")
        if os.path.exists(path):
            with Image.open(path) as im:
                widths.append(im.size[0])
    if not widths:
        return 140
    typical = sorted(widths)[len(widths) // 2]
    return int(min(180, max(110, typical * 2)))


DISPLAY_WIDTH = image_display_width(df["id"].tolist())

cosine_sim = build_similarity(df)
indices = pd.Series(df.index, index=df["id"])     # reverse map, like the practical
item_sim = build_collaborative(df)

st.markdown("<div class='brass-rule'></div>", unsafe_allow_html=True)
st.markdown("<div class='wordmark'>Outfit Recommender</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='tagline'>Upload a piece you own. Three recommender methods each "
    "suggest what completes the look.</div>",
    unsafe_allow_html=True,
)
_res = "low-res" if DISPLAY_WIDTH < 150 else "high-res"
st.markdown(
    f"<div class='count'>{len(df)} products &nbsp;&middot;&nbsp; {_res} images</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='hairline'></div>", unsafe_allow_html=True)

left, right = st.columns([1, 2])

with left:
    st.markdown("<div class='step'>Step 1 &mdash; Your item</div>", unsafe_allow_html=True)

    # Two ways to supply the item. st.camera_input returns the same kind of
    # file-like object as st.file_uploader, so everything after this point
    # is identical for both.
    #
    # camera_input keeps its captured photo across reruns as long as its key
    # doesn't change, so a naive "camera wins if present" check gets stuck
    # on the first photo forever, with no way to switch to a fresh upload.
    # Giving it a key tied to a counter, and incrementing that counter on a
    # "Remove photo" click, makes Streamlit treat it as a brand-new, empty
    # widget - the standard way to reset a widget Streamlit has no built-in
    # clear for.
    if "camera_reset" not in st.session_state:
        st.session_state.camera_reset = 0

    tab_upload, tab_camera = st.tabs(["Upload a photo", "Use camera"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Photo of a top, bottom, dress or shoe", type=["jpg", "jpeg", "png"]
        )

    with tab_camera:
        snapshot = st.camera_input(
            "Point at the item and take a photo",
            key=f"camera_{st.session_state.camera_reset}",
        )
        if snapshot:
            st.caption(
                "Tip: fill the frame with the garment and avoid busy "
                "backgrounds, so the colour is read from the clothing."
            )
            if st.button("Remove photo"):
                st.session_state.camera_reset += 1
                st.rerun()

    # a camera shot takes priority if both are present
    source = snapshot if snapshot is not None else uploaded

    detected = None
    if source:
        photo = Image.open(source)
        st.image(photo, caption="Your item", width='stretch')
        detected, second_colour = detect_colours(photo)
        if detected and second_colour:
            st.success(
                f"Detected colours from photo: **{detected}** and "
                f"**{second_colour}**"
            )
            st.caption(
                "This item appears to be two-tone. Choose below whichever "
                "colour you want the outfit built around."
            )
        elif detected:
            st.success(f"Detected colour from photo: **{detected}**")

    st.markdown("<div class='step'>Step 2 &mdash; Confirm details</div>", unsafe_allow_html=True)
    group_labels = {"Top": "top", "Bottom": "bottom", "Dress": "dress", "Shoes": "shoe"}
    group = st.selectbox("Item type", list(group_labels.keys()))
    selected_slot = group_labels[group]
    # Specific types are read directly from the data rather than a fixed list,
    # since H&M's real product_type_name has far more distinct values than a
    # hand-written list could practically enumerate.
    options = sorted(df[df["slot"] == selected_slot]["articleType"].unique())
    article_type = st.selectbox("Specific type", options)

    # Everyday colour names, not H&M's internal vocabulary. The photo's
    # detected colour pre-selects the closest of these as a starting point,
    # but the user can always override it.
    friendly_names = list(STANDARD_COLOURS.keys())
    if detected:
        default_friendly = min(
            friendly_names,
            key=lambda f: colour_distance(detected, STANDARD_COLOURS[f]),
        )
    else:
        default_friendly = "Black"
    chosen_friendly = st.selectbox(
        "Colour", friendly_names,
        index=friendly_names.index(default_friendly),
        help="Pick the everyday name that best describes your item. If it "
             "broadly agrees with what the photo shows, the more specific "
             "shade detected from the photo is used.",
    )

    # Combine both signals: photo detection (specific) + user choice (reliable)
    colour = resolve_query_colour(detected, STANDARD_COLOURS[chosen_friendly])
    if detected and colour != STANDARD_COLOURS[chosen_friendly]:
        st.caption(f"Using **{colour}** (more specific shade read from your photo).")
    elif detected and colour_distance(detected, STANDARD_COLOURS[chosen_friendly]) > 2.5:
        st.caption(f"Using your choice (**{colour}**) over the photo reading (**{detected}**).")

    gender_options = sorted(df["gender"].unique())
    default_gender = gender_options.index("Women") if "Women" in gender_options else 0
    gender = st.selectbox(
        "Gender", gender_options, index=default_gender,
        help="Not read from the photo - please set this yourself.",
    )
    look = st.selectbox("Dress style", sorted(df["look"].unique()))
    season = st.selectbox("Season", sorted(df["season"].unique()))
    search = st.button("Find matching outfits", type="primary")
    if st.button("Show me another option"):
        st.session_state["variant"] = st.session_state.get("variant", 0) + 1
        search = True
    variant = st.session_state.get("variant", 0)

with right:
    if not search:
        st.markdown(
            "<div class='placeholder'>Set the details on the left, then press "
            "<b>Find matching outfits</b>.<br>All three recommender methods will "
            "be shown side by side for comparison.</div>",
            unsafe_allow_html=True,
        )
    else:
        # A newly uploaded photo is not in the catalogue, so we match it to the
        # closest existing product and recommend from there.
        prefs = {"gender": gender, "season": season, "colour": colour, "look": look}
        allowed_genders = GENDER_GROUPS.get(gender, [gender, "Unisex"])

        pool = df[df["articleType"] == article_type]
        matching_gender = pool[pool["gender"].isin(allowed_genders)]
        if not matching_gender.empty:
            pool = matching_gender

        if pool.empty:
            st.warning("No products of that type in the sample. Try another type.")
        else:
            pool = pool.copy()
            pool["match"] = (
                (pool["look"] == look).astype(float) * 2      # style matters most
                + pool["season"].apply(lambda s: float(season_ok(s, season)))
                + pool["baseColour"].apply(lambda c: colour_score(colour, c))
            )
            anchor_id = pool.sort_values("match", ascending=False).iloc[0]["id"]

            names = df.set_index("id")["productDisplayName"]
            colours = df.set_index("id")["baseColour"]
            types = df.set_index("id")["articleType"]

            methods = [
                ("Content-based", "Item attributes: colour, style, season",
                 lambda: get_recommendations(anchor_id, df, cosine_sim, indices,
                                             prefs, offset=variant)),
                ("Collaborative", "What similar shoppers bought together",
                 lambda: collaborative_recommendations(anchor_id, df, item_sim,
                                                       prefs, offset=variant)),
                ("Hybrid", "Both signals combined, 50/50",
                 lambda: hybrid_recommendations(anchor_id, df, cosine_sim, indices,
                                                item_sim, prefs, offset=variant)),
            ]

            for number, (name, blurb, fn) in enumerate(methods, 1):
                results = fn()

                # Inline styles throughout: class-based CSS can be overridden by
                # Streamlit's theme, and a stylesheet can fail to apply entirely.
                # Inline styles always render, which matters for a live demo.
                st.markdown(
                    "<div style=\"margin:34px 0 14px 0;padding:14px 18px;"
                    "background:#12212E;border-radius:6px;\">"
                    "<div style=\"display:flex;align-items:center;gap:14px;\">"
                    "<span style=\"font-family:'Fraunces',serif;font-size:15px;"
                    "font-weight:700;color:#12212E;background:#C8A06B;"
                    "border-radius:4px;padding:3px 10px;letter-spacing:0.5px;\">"
                    f"{number:02d}</span>"
                    "<span style=\"font-family:'Fraunces',serif;font-size:27px;"
                    "font-weight:700;color:#FFFFFF;letter-spacing:-0.3px;"
                    f"line-height:1.1;\">{name}</span>"
                    "</div>"
                    "<div style=\"font-family:'Inter',sans-serif;font-size:14px;"
                    f"color:#C3CDD6;margin-top:7px;margin-left:2px;\">{blurb}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                if not results:
                    st.warning("No matches with this method. Try different filters.")
                    continue

                columns = st.columns(5)
                for column, result_id in zip(columns, results):
                    path = os.path.join(IMG_DIR, f"{result_id}.jpg")
                    with column:
                        if os.path.exists(path):
                            st.image(path, width=DISPLAY_WIDTH)
                        st.markdown(
                            "<div style=\"font-family:'Inter',sans-serif;font-size:0.79rem;"
                            "font-weight:500;color:#12212E;line-height:1.35;"
                            f"margin-top:0.5rem;\">{names.get(result_id, result_id)}</div>"
                            "<div style=\"font-family:'Inter',sans-serif;font-size:0.71rem;"
                            "color:#93A0AC;margin-top:0.15rem;letter-spacing:0.02em;\">"
                            f"{types.get(result_id, '')} &middot; {colours.get(result_id, '')}</div>",
                            unsafe_allow_html=True,
                        )
