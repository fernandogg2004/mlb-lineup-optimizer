# MLB Lineup Optimizer — Environment Setup

## Prerequisites
- Python 3.11+
- Java 11 or 17 (required by PySpark)
- AWS CLI configured (`aws configure`)

---

## macOS / Linux

```bash
# 1. Create virtual environment
python3.11 -m venv .venv

# 2. Activate
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify Spark + Delta compatibility
python -c "import pyspark; from delta import configure_spark_with_delta_pip; print('OK')"
```

---

## Windows (PowerShell)

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify Spark + Delta compatibility
python -c "import pyspark; from delta import configure_spark_with_delta_pip; print('OK')"
```

> **Note (Windows):** If `Activate.ps1` is blocked by execution policy, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Environment Variables

Copy and populate before running any module:

```bash
cp config/.env.example config/.env
```

Required keys:
```
AWS_DEFAULT_REGION=us-east-1
MLB_S3_BUCKET=mlb-lakehouse
TOMORROW_IO_API_KEY=<your_key>
NOAA_USER_AGENT=mlb-optimizer/1.0 (your@email.com)
KINESIS_WEATHER_STREAM=weather-snapshots
```
