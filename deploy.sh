#!/usr/bin/env bash
# Deploy Maple to GCP Cloud Run.
#
# Usage:
#   PROJECT=your-project-id ./deploy.sh
#
# Two regions matter here, and they're not the same thing:
#   REGION          Where the Cloud Run container itself runs.
#                    Defaults to northamerica-northeast2 (Toronto).
#   VERTEX_LOCATION  Where Vertex AI serves the Gemini model from.
#                    Defaults to global, because the default models
#                    (gemini-3.1-*) are served from the global endpoint only.
#                    That means prompts may be processed outside Canada.
#                    For Canadian-only processing deploy with
#                      VERTEX_LOCATION=northamerica-northeast1
#                      CHAT_MODEL=gemini-2.5-flash PRO_MODEL=gemini-2.5-pro
#                    A 400 FAILED_PRECONDITION or 404 from the model means
#                    the region does not serve that model - check
#                    https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations

set -euo pipefail

PROJECT="${PROJECT:?Set PROJECT=your-gcp-project-id}"
REGION="${REGION:-northamerica-northeast2}"
VERTEX_LOCATION="${VERTEX_LOCATION:-global}"

# Pass the models through only when they are set, so the app defaults apply otherwise.
MODEL_ENV=""
[ -n "${CHAT_MODEL:-}" ] && MODEL_ENV="$MODEL_ENV,CHAT_MODEL=$CHAT_MODEL"
[ -n "${PRO_MODEL:-}" ] && MODEL_ENV="$MODEL_ENV,PRO_MODEL=$PRO_MODEL"
SERVICE="${SERVICE:-maple}"

echo ">> Project:         $PROJECT"
echo ">> Cloud Run region: $REGION"
echo ">> Vertex location:  $VERTEX_LOCATION"
echo ">> Service:         $SERVICE"

gcloud config set project "$PROJECT"

echo ">> Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  aiplatform.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --quiet

echo ">> Building and deploying (Cloud Build will package the container)..."
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --concurrency 40 \
  --min-instances 0 \
  --max-instances 5 \
  --set-env-vars "ENVIRONMENT=prod,LOG_LEVEL=INFO,DEPLOYMENT_REGION=$REGION,GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=$PROJECT,GOOGLE_CLOUD_LOCATION=$VERTEX_LOCATION,REQUIRE_API_KEY=FALSE$MODEL_ENV" \
  --quiet

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')
echo ""
echo ">> Done. Maple is live at: $URL"
echo ">> Try it: curl $URL/catalog"
echo ">> Note: don't curl \$URL/healthz to check this is up - Cloud Run's"
echo ">> edge reserves that literal path and 404s it even when the app is"
echo ">> fine. /catalog or / are the real check."
