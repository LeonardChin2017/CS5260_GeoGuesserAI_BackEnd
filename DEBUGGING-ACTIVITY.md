# Debug: Activity summary not showing

Follow these steps to see where the activity flow breaks.

## 1. Backend: Is the server sending activity steps?

After uploading a resume, check the backend log (e.g. `tail -f backend.log` or PM2 logs):

- You should see: `[RESUME] >>> Sending response: activitySteps count = N` (N ≥ 2).
- Then: `[RESUME]   step 1 : ...`, `step 2 : ...`, etc.

If you **don’t** see this, the backend is not building/sending steps (check for errors earlier in the log).

**Optional:** Call the debug endpoint from the server:

```bash
curl -s http://localhost:3001/api/debug/activity
```

- `backend_activity_count > 0` means activity is being stored.
- Activity steps are sent in **POST /api/resume/upload** (response field `activitySteps`) and in **GET /api/resume** (field `activitySteps`).

## 2. Frontend: Is the upload response received and parsed?

Open the **browser DevTools → Console**, then upload a resume.

You should see, in order:

1. **`[Resume] Activity steps from backend: N [...]`**  
   → Backend returned steps and the frontend received them.

2. **`[Dashboard] activity steps received: N [...]`**  
   → Dashboard received the steps and updated state.

3. **`[AgentActivityPanel] steps: N hasResumeActivity: true ... showing: steps or chat`**  
   → The panel got the steps and is showing them.

If you see **`[Resume] No activitySteps in response`**:

- Backend might not be sending `activitySteps`, or the request failed.
- In **Network** tab: select the `resume/upload` (or upload) request → **Response** → check if the JSON has an `activitySteps` array.

If you see **`[Resume] Backend not configured`**:

- Set `VITE_API_URL` in `Frontend/.env` to your backend URL (e.g. `http://localhost:3001`) and restart the frontend.

## 3. Quick checklist

| Step | Where to check | What you want to see |
|------|----------------|----------------------|
| Backend sends steps | Backend log | `[RESUME] >>> Sending response: activitySteps count = N` |
| Frontend gets steps | Browser console | `[Resume] Activity steps from backend: N` |
| Dashboard gets steps | Browser console | `[Dashboard] activity steps received: N` |
| Panel shows steps | Browser console | `[AgentActivityPanel] steps: N ... showing: steps or chat` |

If step 1 fails → fix backend (errors, port, env).  
If step 2 fails → fix API URL / CORS / auth / upload request.  
If steps 2–3 pass but the UI is still empty → check React state and that `steps` is passed into `AgentActivityPanel`.
