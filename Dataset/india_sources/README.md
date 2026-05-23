India source bundle for the replacement data pipeline.

Files:
- `naukri_data.csv`
- `naukri_processed_data.csv`
- `poova_data_analyst.csv`
- `poova_data_scientist.csv`

Why these are here:
- They are public India-focused job data sources that can be downloaded without depending on the old Pakistan dataset.
- `scripts/prepare_india_dataset.py` standardizes them into a single schema and combines them with the fraud base from `fake_job_postings.csv`.

Notes:
- The Naukri mirror is a public GitHub mirror of the PromptCloud sample.
- The Poova files are India-specific Naukri scrapes for analyst/scientist roles.
- The live Flask app still runs on the existing saved model artifacts; these files are for India-focused dataset preparation and future retraining.
