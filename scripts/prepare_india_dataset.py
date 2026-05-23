from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile
from urllib.request import urlretrieve

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "Dataset"
SOURCES_DIR = DATASET_DIR / "india_sources"

EMSCAD_PATH = DATASET_DIR / "fake_job_postings.csv"
TEST_DATA_PATH = DATASET_DIR / "testData.csv"
LEGACY_TEST_DATA_PATH = DATASET_DIR / "testData_legacy_global.csv"

REAL_OUTPUT_PATH = DATASET_DIR / "india_real_jobs.csv"
SYNTHETIC_OUTPUT_PATH = DATASET_DIR / "india_synthetic_fraud.csv"
TRAINING_OUTPUT_PATH = DATASET_DIR / "india_training_dataset.csv"
SUMMARY_OUTPUT_PATH = DATASET_DIR / "india_dataset_summary.json"

SOURCE_FILES = {
    "naukri_data.csv": "https://raw.githubusercontent.com/mrmaheshrajput/edanaukri/master/data.csv",
    "naukri_processed_data.csv": "https://raw.githubusercontent.com/mrmaheshrajput/edanaukri/master/processed_data.csv",
    "poova_data_analyst.csv": "https://raw.githubusercontent.com/Poova53/Analysis-of-data-scientist-and-data-analyst-job-in-India/master/1.%20Scraping%20data/data%20analyst.csv",
    "poova_data_scientist.csv": "https://raw.githubusercontent.com/Poova53/Analysis-of-data-scientist-and-data-analyst-job-in-India/master/1.%20Scraping%20data/data%20scientist.csv",
}

INDIA_CITY_FALLBACKS = [
    "Bengaluru",
    "Hyderabad",
    "Mumbai",
    "Pune",
    "Delhi NCR",
    "Chennai",
    "Noida",
    "Gurugram",
]

SCAM_TEMPLATES = [
    (
        "Urgent hiring for {title} in {location}. Selected applicants can start immediately "
        "after a short WhatsApp screening. Compensation is {salary}. To confirm the interview "
        "slot, share Aadhaar, PAN, and a refundable registration fee over UPI today."
    ),
    (
        "Premium placement partner is onboarding candidates for {title} across {location}. "
        "The company guarantees fast joining, flexible work from home, and salary up to {salary}. "
        "Candidates must pay the document verification charge before the final HR round."
    ),
    (
        "Immediate openings for {title}. Interview rounds are handled only on Telegram for faster "
        "processing. The recruiter will share the offer letter once the security deposit is paid. "
        "Preferred location is {location} and the expected package is {salary}."
    ),
    (
        "Government and private sector hiring drive for {title} in {location}. No hard skills test "
        "is required. Shortlisted applicants should send bank details and a training fee to reserve "
        "their batch. Salary offer ranges around {salary}."
    ),
]

SCAM_REQUIREMENTS = [
    "Share Aadhaar, PAN, and bank account details before onboarding.",
    "Pay the registration fee by UPI to activate the joining workflow.",
    "WhatsApp and Telegram communication only; no official company email required.",
    "Interview confirmation depends on a refundable document verification charge.",
]

STANDARD_COLUMNS = [
    "job_id",
    "title",
    "company_name",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "fraudulent",
    "label",
    "source",
    "source_url",
    "is_indian_market",
]

POOVA_SKILL_COLUMNS = ["python", "r", "sql", "excel", "vba", "powerbi", "tableau", "nosql", "sas", "git", "matlab"]


def safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_location(value: object) -> str:
    location = safe_text(value)
    location = location.replace("View More", "")
    location = location.replace("/Secunderabad", "")
    location = re.sub(r"\s*,\s*", ", ", location)
    location = re.sub(r"\s+", " ", location).strip(" ,")
    return location or random.choice(INDIA_CITY_FALLBACKS)


def normalize_key(*parts: object) -> str:
    text = " ".join(safe_text(part).lower() for part in parts if safe_text(part))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def download_sources(force: bool = False) -> None:
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCE_FILES.items():
        target = SOURCES_DIR / filename
        if target.exists() and not force:
            continue
        print(f"Downloading {filename} from {url}")
        urlretrieve(url, target)


def standard_row(**kwargs: object) -> dict[str, object]:
    row = {
        "job_id": "",
        "title": "",
        "company_name": "",
        "location": "",
        "department": "",
        "salary_range": "",
        "company_profile": "",
        "description": "",
        "requirements": "",
        "benefits": "",
        "telecommuting": 0,
        "has_company_logo": 0,
        "has_questions": 0,
        "employment_type": "",
        "required_experience": "",
        "required_education": "",
        "industry": "",
        "function": "",
        "fraudulent": 0,
        "label": 0,
        "source": "",
        "source_url": "",
        "is_indian_market": True,
    }
    row.update(kwargs)
    row["fraudulent"] = int(row["fraudulent"])
    row["label"] = int(row["fraudulent"])
    return row


def skill_list_from_text(value: object) -> str:
    raw = safe_text(value)
    if not raw:
        return ""
    chunks = [part.strip() for part in re.split(r"[|,]", raw) if part.strip()]
    return ", ".join(chunks)


def build_naukri_description(row: pd.Series) -> str:
    title = safe_text(row["Job Title"])
    location = clean_location(row["Location"])
    experience = safe_text(row["Job Experience Required"])
    salary = safe_text(row["Job Salary"])
    skills = skill_list_from_text(row["Key Skills"])
    role_category = safe_text(row["Role Category"])
    function = safe_text(row["Functional Area"])
    industry = safe_text(row["Industry"])
    role = safe_text(row["Role"])
    parts = [
        f"{title} opportunity in {location}.",
        f"Experience required: {experience or 'Not specified'}.",
        f"Compensation: {salary or 'Not disclosed'}.",
        f"Key skills: {skills or 'Not specified'}.",
        f"Role category: {role_category or 'Not specified'}.",
        f"Functional area: {function or 'Not specified'}.",
        f"Industry: {industry or 'Not specified'}.",
        f"Role focus: {role or 'Not specified'}.",
        "This posting comes from an India job-market feed collected from Naukri listings.",
    ]
    return " ".join(parts)


def load_naukri_real_jobs() -> pd.DataFrame:
    df = pd.read_csv(SOURCES_DIR / "naukri_data.csv").fillna("")
    rows = []
    for index, record in df.iterrows():
        rows.append(
            standard_row(
                job_id=f"naukri_{safe_text(record['Uniq Id']) or index + 1}",
                title=safe_text(record["Job Title"]),
                location=clean_location(record["Location"]),
                department=safe_text(record["Role Category"]),
                salary_range=safe_text(record["Job Salary"]),
                description=build_naukri_description(record),
                requirements=skill_list_from_text(record["Key Skills"]),
                required_experience=safe_text(record["Job Experience Required"]),
                industry=safe_text(record["Industry"]),
                function=safe_text(record["Functional Area"]),
                source="naukri_30k_mirror",
                source_url=SOURCE_FILES["naukri_data.csv"],
            )
        )
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def poova_skills(record: pd.Series) -> str:
    skills = [column for column in POOVA_SKILL_COLUMNS if safe_text(record.get(column))]
    return ", ".join(skills)


def build_poova_description(record: pd.Series, track: str) -> str:
    title = safe_text(record["job title"])
    company = safe_text(record["company"])
    location = clean_location(record["location"])
    experience = safe_text(record["experience"])
    salary = safe_text(record["salary"])
    education = safe_text(record["education"])
    skills = poova_skills(record)
    return (
        f"{title} role in {location} with {company or 'an India-based employer'}. "
        f"Track: {track}. Experience: {experience or 'Not specified'}. "
        f"Compensation: {salary or 'Not disclosed'}. "
        f"Preferred education: {education or 'Not specified'}. "
        f"Core skills: {skills or 'data analysis and communication'}. "
        "This listing was collected from India-focused Naukri scraping for analytics roles."
    )


def load_poova_real_jobs() -> pd.DataFrame:
    source_specs = [
        ("poova_data_analyst.csv", "Data Analyst"),
        ("poova_data_scientist.csv", "Data Scientist"),
    ]
    rows = []
    for filename, track in source_specs:
        df = pd.read_csv(SOURCES_DIR / filename).fillna("")
        for index, record in df.iterrows():
            rows.append(
                standard_row(
                    job_id=f"{track.lower().replace(' ', '_')}_{index + 1}",
                    title=safe_text(record["job title"]),
                    company_name=safe_text(record["company"]),
                    location=clean_location(record["location"]),
                    salary_range=safe_text(record["salary"]),
                    description=build_poova_description(record, track),
                    requirements=poova_skills(record),
                    required_experience=safe_text(record["experience"]),
                    required_education=safe_text(record["education"]),
                    industry="Data Science and Analytics",
                    function=track,
                    source=f"poova_{track.lower().replace(' ', '_')}",
                    source_url=SOURCE_FILES[filename],
                )
            )
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def load_emscad_fraud_jobs() -> pd.DataFrame:
    df = pd.read_csv(EMSCAD_PATH).fillna("")
    fraud_rows = df[df["fraudulent"].astype(int) == 1].copy()
    rows = []
    for _, record in fraud_rows.iterrows():
        rows.append(
            standard_row(
                job_id=safe_text(record["job_id"]),
                title=safe_text(record["title"]),
                location=safe_text(record["location"]),
                department=safe_text(record["department"]),
                salary_range=safe_text(record["salary_range"]),
                company_profile=safe_text(record["company_profile"]),
                description=safe_text(record["description"]),
                requirements=safe_text(record["requirements"]),
                benefits=safe_text(record["benefits"]),
                telecommuting=int(record["telecommuting"]) if safe_text(record["telecommuting"]) else 0,
                has_company_logo=int(record["has_company_logo"]) if safe_text(record["has_company_logo"]) else 0,
                has_questions=int(record["has_questions"]) if safe_text(record["has_questions"]) else 0,
                employment_type=safe_text(record["employment_type"]),
                required_experience=safe_text(record["required_experience"]),
                required_education=safe_text(record["required_education"]),
                industry=safe_text(record["industry"]),
                function=safe_text(record["function"]),
                fraudulent=1,
                source="emscad_fraud_base",
                source_url="https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction",
                is_indian_market=False,
            )
        )
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def load_emscad_india_demo_jobs() -> pd.DataFrame:
    df = pd.read_csv(EMSCAD_PATH).fillna("")
    india_rows = df[df["location"].astype(str).str.startswith("IN,")].copy()
    rows = []
    for _, record in india_rows.iterrows():
        rows.append(
            standard_row(
                job_id=safe_text(record["job_id"]),
                title=safe_text(record["title"]),
                location=safe_text(record["location"]),
                department=safe_text(record["department"]),
                salary_range=safe_text(record["salary_range"]),
                company_profile=safe_text(record["company_profile"]),
                description=safe_text(record["description"]),
                requirements=safe_text(record["requirements"]),
                benefits=safe_text(record["benefits"]),
                telecommuting=int(record["telecommuting"]) if safe_text(record["telecommuting"]) else 0,
                has_company_logo=int(record["has_company_logo"]) if safe_text(record["has_company_logo"]) else 0,
                has_questions=int(record["has_questions"]) if safe_text(record["has_questions"]) else 0,
                employment_type=safe_text(record["employment_type"]),
                required_experience=safe_text(record["required_experience"]),
                required_education=safe_text(record["required_education"]),
                industry=safe_text(record["industry"]),
                function=safe_text(record["function"]),
                fraudulent=int(record["fraudulent"]),
                source="emscad_india_demo",
                source_url="https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction",
                is_indian_market=True,
            )
        )
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def default_salary(value: str) -> str:
    value = safe_text(value)
    if value and value.lower() != "not disclosed":
        return value
    return random.choice(["3-6 LPA", "4-8 LPA", "5-10 LPA", "6-12 LPA"])


def build_india_scam_jobs(real_jobs: pd.DataFrame, count: int) -> pd.DataFrame:
    if real_jobs.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    sample_size = min(count, len(real_jobs))
    sampled = real_jobs.sample(n=sample_size, random_state=42).reset_index(drop=True)
    rows = []
    for index, record in sampled.iterrows():
        title = safe_text(record["title"]) or "Back Office Executive"
        location = clean_location(record["location"])
        salary = default_salary(safe_text(record["salary_range"]))
        company_name = safe_text(record["company_name"]) or "Confidential Placement Partner"
        scam_template = SCAM_TEMPLATES[index % len(SCAM_TEMPLATES)]
        description = scam_template.format(title=title, location=location, salary=salary)
        requirements = " ".join(
            [
                SCAM_REQUIREMENTS[index % len(SCAM_REQUIREMENTS)],
                safe_text(record["requirements"]) or "Communication, Excel, and immediate joining.",
            ]
        )
        rows.append(
            standard_row(
                job_id=f"india_scam_{index + 1}",
                title=f"Urgent Hiring - {title}",
                company_name=company_name,
                location=location,
                department=safe_text(record["department"]) or "Recruitment",
                salary_range=salary,
                company_profile="Fast-track recruiting partner claiming India-wide urgent openings.",
                description=description,
                requirements=requirements,
                employment_type=safe_text(record["employment_type"]) or "Full-time",
                required_experience=safe_text(record["required_experience"]) or "0-5 years",
                required_education=safe_text(record["required_education"]) or "Any Graduate",
                industry=safe_text(record["industry"]) or "Staffing and Recruiting",
                function=safe_text(record["function"]) or "Operations",
                fraudulent=1,
                source="synthetic_india_scam",
                source_url="local_rule_based_generation",
                is_indian_market=True,
            )
        )
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def deduplicate_jobs(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["_key"] = working.apply(
        lambda row: normalize_key(row["title"], row["location"], row["description"]), axis=1
    )
    working = working.drop_duplicates(subset="_key", keep="first")
    return working.drop(columns="_key").reset_index(drop=True)


def write_summary(real_jobs: pd.DataFrame, fraud_jobs: pd.DataFrame, training_jobs: pd.DataFrame) -> None:
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "real_jobs": int(len(real_jobs)),
            "fraud_jobs": int(len(fraud_jobs)),
            "training_rows": int(len(training_jobs)),
            "fraud_ratio": round((training_jobs["fraudulent"].mean() * 100), 2) if len(training_jobs) else 0.0,
        },
        "sources": training_jobs.groupby(["source", "fraudulent"]).size().reset_index(name="rows").to_dict(orient="records"),
        "files": {
            "real_jobs": str(REAL_OUTPUT_PATH.name),
            "synthetic_fraud": str(SYNTHETIC_OUTPUT_PATH.name),
            "training_dataset": str(TRAINING_OUTPUT_PATH.name),
            "default_demo_dataset": str(TEST_DATA_PATH.name),
        },
    }
    SUMMARY_OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_demo_dataset() -> None:
    if TEST_DATA_PATH.exists() and not LEGACY_TEST_DATA_PATH.exists():
        copyfile(TEST_DATA_PATH, LEGACY_TEST_DATA_PATH)

    demo_pool = deduplicate_jobs(load_emscad_india_demo_jobs())
    demo_real = demo_pool[demo_pool["fraudulent"] == 0].sample(
        n=min(16, len(demo_pool[demo_pool["fraudulent"] == 0])),
        random_state=7,
    )
    demo_fraud = demo_pool[demo_pool["fraudulent"] == 1].sample(
        n=min(4, len(demo_pool[demo_pool["fraudulent"] == 1])),
        random_state=11,
    )
    demo = pd.concat([demo_real, demo_fraud], ignore_index=True)
    demo = demo.sample(frac=1.0, random_state=23).reset_index(drop=True)
    demo[STANDARD_COLUMNS].to_csv(TEST_DATA_PATH, index=False)


def build_datasets(synthetic_fraud_count: int) -> dict[str, pd.DataFrame]:
    naukri_jobs = load_naukri_real_jobs()
    poova_jobs = load_poova_real_jobs()
    real_jobs = deduplicate_jobs(pd.concat([naukri_jobs, poova_jobs], ignore_index=True))

    fraud_base = deduplicate_jobs(load_emscad_fraud_jobs())
    synthetic_fraud = deduplicate_jobs(build_india_scam_jobs(real_jobs, synthetic_fraud_count))
    fraud_jobs = deduplicate_jobs(pd.concat([fraud_base, synthetic_fraud], ignore_index=True))

    training_jobs = deduplicate_jobs(pd.concat([real_jobs, fraud_jobs], ignore_index=True))
    return {
        "real_jobs": real_jobs,
        "synthetic_fraud": synthetic_fraud,
        "training_jobs": training_jobs,
        "fraud_jobs": fraud_jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an India-focused job fraud dataset bundle.")
    parser.add_argument("--refresh-downloads", action="store_true", help="Re-download the public India source files.")
    parser.add_argument("--synthetic-fraud-count", type=int, default=2500, help="Number of rule-based India scam rows to generate.")
    args = parser.parse_args()

    random.seed(42)
    download_sources(force=args.refresh_downloads)
    bundles = build_datasets(args.synthetic_fraud_count)

    REAL_OUTPUT_PATH.write_text("", encoding="utf-8")
    bundles["real_jobs"][STANDARD_COLUMNS].to_csv(REAL_OUTPUT_PATH, index=False)
    bundles["synthetic_fraud"][STANDARD_COLUMNS].to_csv(SYNTHETIC_OUTPUT_PATH, index=False)
    bundles["training_jobs"][STANDARD_COLUMNS].to_csv(TRAINING_OUTPUT_PATH, index=False)
    write_demo_dataset()
    write_summary(bundles["real_jobs"], bundles["fraud_jobs"], bundles["training_jobs"])

    print(f"Wrote {REAL_OUTPUT_PATH.name} with {len(bundles['real_jobs']):,} real rows")
    print(f"Wrote {SYNTHETIC_OUTPUT_PATH.name} with {len(bundles['synthetic_fraud']):,} synthetic fraud rows")
    print(f"Wrote {TRAINING_OUTPUT_PATH.name} with {len(bundles['training_jobs']):,} total rows")
    print(f"Updated {TEST_DATA_PATH.name} with India-focused demo samples")


if __name__ == "__main__":
    main()
