# Online deployment template

The local Docker Compose deployment is the canonical reproducible setup. The cloud setup below is a template; it requires the owner to create accounts and set secrets in the provider dashboards.

## Recommended free layout

- Frontend: Vercel, serving `app/frontend/site`.
- Backend and AI service: Render Docker web services.
- Database: MongoDB Atlas free cluster.
- Demo tunnel: ngrok only for local demonstrations; it is not a production deployment.

## Required cloud variables

Never commit real values. Configure these in Render/Vercel dashboards:

```text
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>/<database>?retryWrites=true&w=majority
MONGODB_DATABASE=customer_churn
AI_SERVICE_URL=https://<ai-service>.onrender.com
BACKEND_URL=https://<backend>.onrender.com
```

The MongoDB Atlas Network Access rule must allow the selected Render outbound addresses or the provider's documented range. Use a database user with only the required database permissions.

## Render

`render.yaml` describes the AI and backend Docker services. After creating the Blueprint, set `MONGODB_URI`, `MONGODB_DATABASE` and `AI_SERVICE_URL` as secret/environment values in Render. Verify `/health` for both services before connecting the frontend.

## Vercel

Set the Vercel project Root Directory to `app/frontend/site`. The included
`vercel.json` rewrites `/api/*` to the backend. Replace the placeholder backend
hostname before deployment. The browser must call the backend through the
rewrite; users should never call the AI service directly.

## Local fallback and ngrok

```powershell
docker compose up --build -d
ngrok http 3000
```

The ngrok process must remain open. Its URL changes or stops when the process or machine stops. This is suitable for a weekly demo, not uptime-sensitive hosting.

## Verification checklist

1. `GET /health` on backend reports MongoDB and AI healthy.
2. `GET /health` on AI reports all model artifacts loaded.
3. `GET /api/models` lists the four models.
4. Call each `/api/predict/<model>` route with a valid 20-feature payload.
5. Check `docker compose logs backend ai-service` and match the `X-Request-ID` through both services.
6. Confirm no `.env`, token or password is committed before pushing.
