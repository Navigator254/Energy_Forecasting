# Deployment Secrets

## Local
Use `.env` in the project root:


EIA_API_KEY="RVTxwhBOtSN6ndMp7vD5ezknLBc7eX1W0ZJyEc1x "                                    EIA_PARENT_REGION=CISO
EIA_SUBBA=
EIA_MIN_VALID_MW=1000

## GitHub Actions
Add this repository secret:
- EIA_API_KEY

## Streamlit Community Cloud
In App Settings -> Secrets, add:


EIA_API_KEY="RVTxwhBOtSN6ndMp7vD5ezknLBc7eX1W0ZJyEc1x "          EIA_PARENT_REGION="CISO"
EIA_SUBBA=""
EIA_MIN_VALID_MW="1000"
