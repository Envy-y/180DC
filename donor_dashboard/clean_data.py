import pandas as pd
import numpy as np

def clean_data(data):
    # variable of interests
    variables = [
        "Q289",  # Religion/denomination: 0 = No denomination; 1 = Roman Catholic; 2 = Protestant; 3 = Orthodox; 4 = Jew; 5 = Muslim; 6 = Hindu; 7 = Buddhist; 8 = Other 
        "Q260",  # Sex of respondent: 1 = Male; 2 = Female 
        "Q261",  # Year of birth (4-digit), numeric – derive exact age
        "Q262",  # Recorded age in years, numeric
        "Q275",  # Highest educational level (ISCED): 0 Early-childhood; 1 Primary; 2 Lower-secondary; 3 Upper-secondary; 4 Post-secondary non-tertiary; 5 Short-cycle tertiary; 6 Bachelor; 7 Master; 8 Doctorate 
        "Q288",  # Household-income decile: 1 (lowest) … 10 (highest) 
        "Q286",  # Past-year household finances: 1 Saved money; 2 Just get by; 3 Spent some savings; 4 Spent savings & borrowed 
        "Q287",  # Self-perceived social class: 1 Upper; 2 Upper-middle; 3 Lower-middle; 4 Working; 5 Lower 
        "Q273",  # Marital status: 1 Married; 2 Living together; 3 Divorced; 4 Separated; 5 Widowed; 6 Single 
        "Q270",  # Household size (open numeric)
        "H_URBRURAL",  # Urban/rural residence: 1 Urban; 2 Rural 
        "N_REGION_ISO",  # ISO-3166-2 state code where interview was conducted 

        # --- Giving behaviour & attitudes ---
        
        "Q213",  # Donated to a group/campaign: 1 Have done; 2 Might do; 3 Would never do 
        "Q81",   # Confidence in charitable/humanitarian orgs.: 1 Great deal; 2 Quite a lot; 3 Not very much; 4 None 
        "Q83",   # Confidence in the United Nations: same 4-point scale as Q81
        "Q259",  # Closeness felt to the WORLD: 1 Very close; 2 Close; 3 Not very close; 4 Not close at all 
        "Q57",   # Generalised trust: 1 Most people can be trusted; 2 Need to be very careful 
        "Q62", # Trust in people of another religion (1-4 scale): 1 Trust a lot; 2 Trust some; 3 Do not trust much; 4 Do not trust at all 
        "Q64", # Confidence in churches (e.g. mosque,temples) (1-4 scale): 1 Trust a lot; 2 Trust some; 3 Do not trust much; 4 Do not trust at all 
        "Q112", #corrpution (1-10),1 is no corruption, 10 is high corruption 
    
        # Optional cause-specific memberships (interest in types of causes) –
        "Q101",  # Membership in humanitarian/charitable org.: 2 Active; 1 Inactive; 0 Not a member 
        "Q94",   # Church/religious org. membership: 2 Active; 1 Inactive; 0 None 
    ]


    fd = data[variables].copy()


    age_bins = [0, 28, 44, 60, 79, 100]
    labels = ["Gen Z", "Millennials", "Gen X", "Baby Boomers", "Silent Generation"]
    fd['Age Group'] = pd.cut(fd['Q262'], bins=age_bins, labels=labels, right=False)


    fd["Gender"] = fd["Q260"].map({1: "Male", 2: "Female"})

    education_bins =[-6,0,2,4,6,7,9]
    education_labels = ["No response", "Primary or below", "Secondary", "Post-Secondary", "Bachelor", "Postgraduate"]
    fd['Education Level'] = pd.cut(fd['Q275'], bins=education_bins, labels=education_labels, right=False)


    income_bins = [-6, 0, 3, 7, 10]  
    income_labels = ["No Response", "Low Income", "Middle Income", "High Income"]
    fd['Income Level'] = pd.cut(fd['Q288'], bins=income_bins, labels=income_labels, right=False)

    fd["Gender"] = fd["Q260"].map({1: "Male", 2: "Female"})

    fd = fd[fd["Q213"].isin([1, 2, 3])].copy()
    fd["Potential_Donor"] = fd["Q213"].map({1: 1, 2: 1, 3: 0}).astype(int)

    def flip_1to4(s: pd.Series) -> pd.Series:
        """Flip 1..4 scale so higher = more of the positive construct."""
        s = pd.to_numeric(s, errors="coerce")
        s = s.replace(-1, np.nan)
        return 5 - s  # 1→4, 2→3, 3→2, 4→1

    def first_series(col):
        """Handle duplicate-named columns that come in as a DataFrame (e.g., Q260)."""
        obj = fd.get(col)
        if isinstance(obj, pd.DataFrame):
            return pd.to_numeric(obj.iloc[:, 0], errors="coerce")
        return pd.to_numeric(obj, errors="coerce")

    CURRENT_YEAR = 2025

    # Age / Age Group (prefer year of birth Q261; fallback to recorded age Q262)
    birth_year = pd.to_numeric(fd.get("Q261"), errors="coerce")
    age_rec    = pd.to_numeric(fd.get("Q262"), errors="coerce")
    fd["Age"]  = np.where(birth_year.notna(), CURRENT_YEAR - birth_year, age_rec)

    # Age Group (generational cohorts)
    cohort_bins   = [1900, 1945, 1964, 1980, 1996, 2012, 2026]
    cohort_labels = ["Silent", "Boomer", "Gen X", "Millennial", "Gen Z", "Gen Alpha"]
    fd["Age Group"] = pd.cut(birth_year, bins=cohort_bins, labels=cohort_labels, right=True)
    # Fallback to simple age bands if birth year missing
    age_band = pd.cut(fd["Age"], bins=[-1, 17, 24, 39, 64, 200],
                    labels=["<18", "18-24", "25-39", "40-64", "65+"], right=True)
    fd["Age Group"] = fd["Age Group"].astype("object").where(fd["Age Group"].notna(), age_band)

    # Gender (Q260: 1=Male, 2=Female; handle duplicates)
    fd["Gender"] = first_series("Q260").map({1: "Male", 2: "Female"}).astype("object")

    # Education (Q275 → collapsed bins; include -1)
    edu_map = {
        -1: "No Response",
        0: "Primary or below", 1: "Primary or below",
        2: "Secondary",        3: "Secondary",
        4: "Post-secondary / Diploma", 5: "Post-secondary / Diploma",
        6: "Bachelor",
        7: "Postgraduate",     8: "Postgraduate"
    }
    fd["Education Level"] = pd.to_numeric(fd.get("Q275"), errors="coerce").map(edu_map).astype("object")

    # Income (Q288 decile → Low/Middle/High + No Response)
    fd["Q288"] = pd.to_numeric(fd.get("Q288"), errors="coerce")
    income_bins   = [-2, 0, 3, 7, 10]  # [-1], 1–3, 4–7, 8–10
    income_labels = ["No Response", "Low Income", "Middle Income", "High Income"]
    fd["Income Level"] = pd.cut(fd["Q288"], bins=income_bins, labels=income_labels).astype("object")

    # Religion (Q289; India-specific: 9= Sikh)
    religion_map = {
        -1: "No Response",
        0: "No denomination",
        1: "Christian (Catholic)",
        4: "Jew",
        5: "Muslim",
        6: "Hindu",
        7: "Buddhist",
        8: "Other",
        9: "Other (e.g. Sikh)"   # India-specific category often present in WVS7 country file
    }
    fd["Religion"] = pd.to_numeric(fd.get("Q289"), errors="coerce").map(religion_map).astype("object")

    # Urban / Rural (H_URBRURAL: 1 Urban; 2 Rural)
    fd["Urban/Rural"] = pd.to_numeric(fd.get("H_URBRURAL"), errors="coerce").map({1: "Urban", 2: "Rural"}).astype("object")

    # Marital status (Q273)
    marital_map = {
        -1: "No Response",
        1: "Married",
        2: "Living together",
        3: "Divorced",
        4: "Separated",
        5: "Widowed",
        6: "Single"
    }
    fd["Marital Status"] = pd.to_numeric(fd.get("Q273"), errors="coerce").map(marital_map).astype("object")

    # Self-perceived social class (Q287)
    class_map = {
        -1: "No Response",
        1: "Upper",
        2: "Upper-middle",
        3: "Lower-middle",
        4: "Working",
        5: "Lower"
    }
    fd["Social Class"] = pd.to_numeric(fd.get("Q287"), errors="coerce").map(class_map).astype("object")

    # Household finances (Q286)
    fin_map = {
        -1: "No Response",
        1: "Saved money",
        2: "Just get by",
        3: "Spent some savings",
        4: "Spent savings & borrowed"
    }
    fd["Household Finances"] = pd.to_numeric(fd.get("Q286"), errors="coerce").map(fin_map).astype("object")

    # Family size (Q270) 
    q270   = pd.to_numeric(fd.get("Q270"), errors="coerce")
    def bin_household_size(x):
        if pd.isna(x) or x < 0:        return "No Response"
        elif x <= 2:              return "Small (1-2)"
        elif 3 <= x <= 5:        return "Medium (3-5)"
        elif 6 <= x <= 8:            return "Large (6-8)"
        elif x >= 9:                return "Very large (9+)"
        return "No Response"
    fd["Household Size"] = q270.apply(bin_household_size).astype("object")

    # Region (N_REGION_ISO) – keep as string (you can map to state names later)
    fd["Region"] = fd.get("N_REGION_ISO").astype(str)

    # Q81, Q83, Q64, Q62, Q259 are 1..4 (1=high) → flip
    fd["Trust_Charities"]     = flip_1to4(fd.get("Q81"))
    fd["Trust_UN"]            = flip_1to4(fd.get("Q83"))
    fd["Trust_Churches"]      = flip_1to4(fd.get("Q64"))
    fd["Trust_OtherReligion"] = flip_1to4(fd.get("Q62"))
    fd["Closeness_World"]     = flip_1to4(fd.get("Q259"))

    # Generalized trust Q57: 1=trust, 2=careful
    fd["Trust_General"] = pd.to_numeric(fd.get("Q57"), errors="coerce").replace({-1: np.nan, 1: 1, 2: 0})

    # Corruption Q112: 1 (no corruption) .. 10 (high)
    q112 = pd.to_numeric(fd.get("Q112"), errors="coerce")
    def bin_corruption(x):
        if pd.isna(x) or x == -1:        return "No Response"
        if 1 <= x <= 3:                  return "Low corruption"
        if 4 <= x <= 6:                  return "Medium corruption"
        if 7 <= x <= 10:                 return "High corruption"
        return "No Response"
    fd["Perceived_Corruption"] = q112.apply(bin_corruption).astype("object")

    # Q101 (humanitarian org): 2 Active; 1 Inactive; 0 None; (-1 missing)
    mem_map = {-1: "No Response", 0: "None", 1: "Inactive", 2: "Active"}
    fd["Charity_Membership"] = pd.to_numeric(fd.get("Q101"), errors="coerce").map(mem_map).astype("object")

    # Q94 (religious org): 2 Active; 1 Inactive; 0 None
    fd["ReligiousOrg_Membership"] = pd.to_numeric(fd.get("Q94"), errors="coerce").map(mem_map).astype("object")

    fd["Potential_Donor_Str"] = fd["Potential_Donor"].map({1: "Yes", 0: "No"})

    fd["Region"] = fd["Region"].astype("int64")
    iso_to_state = { 356004: "Bihar", 356008: "Haryana", 356015: "Maharashtra", 356021: "Punjab", 356025: "Andhra Pradesh", 356029: "West Bengal", 356028: "Uttar Pradesh", 356034: "Delhi", }
    fd["State"] = fd["Region"].map(iso_to_state)

    return fd
