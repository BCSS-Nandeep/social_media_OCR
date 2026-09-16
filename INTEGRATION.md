# Integrating OCR extraction into saga-police

Written for whoever wires this OCR service into `saga-police`'s existing
post-ingestion pipeline. Everything below was read directly out of that
repo (`/home/ashish-ratna/saga-police/backend`) — file paths, field names,
and the "what's already there vs. what's missing" table are from the actual
code, not assumed. For the OCR service's own endpoint contract, see
[API.md](API.md); for how it's deployed/operated, see [DEPLOYMENT.md](DEPLOYMENT.md).
This document is the bridge between the two: where in saga-police to call
it, and what to do with the result.

---

## 1. The two services, in one sentence each

- **This OCR service** (`social_media_OCR`, this repo): give it an image
  (URL or base64), it gives back the text on it. One endpoint,
  `POST /extract`. Runs on the `acb` GPU box, currently `http://98.86.63.69:8000`.
- **saga-police** (`backend/`): a multi-tenant Node/Express app that polls
  Instagram, Facebook, Telegram, X and YouTube on a schedule, upserts posts
  into Postgres (`social_media_posts`), and runs sentiment analysis on
  their `text`. This doc adds a third step: extract text baked into the
  post's *image*, not just its caption.

## 2. How saga-police fetches posts today

Identical shape for all five platforms — this is the actual pipeline, read
from the code:

```
scheduler.js  (setInterval, per platform, e.g. INSTAGRAM_MONITOR_TICK_MS)
  -> runProfile.js: runInstagramProfile(accountId, ...)
       -> fetch.js: fetchInstagramPosts(account, auth)   // hits the platform API
            -> blugate.instagram.helpers.js: mapFeedItemToUpsert(node, ...)
                 // raw platform payload -> { platform, external_id, text, media_urls, raw_data, ... }
       -> for (const post of posts) { await upsertPost(post, { db, dbName }) }
            -> upsertPost.js writes social_media_posts, and if `text` changed,
               calls enqueuePost() -> sentiment analysis picks it up automatically
```

Same files exist per platform under
`backend/src/services/monitoringsocialmedia/{instagram,facebook,telegram,x,youtube}/`
(`fetch.js`, `runProfile.js`, `scheduler.js`). This doc uses Instagram as
the worked example; the same change applies to each of the other four at
the equivalent line.

### The row shape every platform maps into (`upsertPost.js`)

```js
{
  account_id, platform, external_id, url,
  text,          // caption/message text -- NOT what's drawn on an image
  author_name, author_handle, media_type,
  media_urls,    // <-- what this integration needs populated
  engagement, posted_at, raw_data,   // raw_data: the full untouched platform payload
}
```

`social_media_posts` (Postgres, `prisma/tenant.schema.prisma`) has no OCR
column today. `raw_data` is a `Json` field with no fixed shape — the
zero-migration place to put an OCR result (§5).

## 3. `media_urls` per platform — checked in the actual mapper code, not assumed

| Platform | File | Status |
|---|---|---|
| Telegram | `blugate.telegram.helpers.js` | **Already populated** — real URLs from `msg.media[].url/src/file_url` |
| X | `monitoringsocialmedia/x/fetch.js` | **Already populated** — `media_url_https` from `extended_entities.media` |
| YouTube | `monitoringsocialmedia/youtube/fetch.js` | **Populated, but it's a thumbnail**, not a post photo — `snippet.thumbnails` |
| Instagram | `blugate.instagram.helpers.js` (`mapFeedItemToUpsert`) | **Hardcoded `media_urls: []`** — not populated |
| Facebook | `monitoringsocialmedia/facebook/fetch.js` | **Hardcoded `media_urls: []`** — not populated |

Instagram and Facebook posts already carry the real image URL in
`raw_data` — it's just not surfaced into `media_urls`. Proof: this
codebase already extracts it elsewhere, for the Global Search preview
path, in `blugate.instagram.helpers.js`'s `mapNodesToSearchPosts`:

```js
const image =
  node.image_versions2?.candidates?.[0]?.url ||
  node.display_uri ||
  node.thumbnail_url ||
  null;
```

**To close the Instagram gap**, add the same picker into
`mapFeedItemToUpsert` (same file) before its `return`:

```js
const image =
  node.image_versions2?.candidates?.[0]?.url ||
  node.display_uri ||
  node.thumbnail_url ||
  null;
// ...
media_urls: image ? [image] : [],
```

**Facebook's raw field name wasn't found in the code inspected for this
doc** (nothing under `blugate.facebook.*` obviously exposes a post image —
only an unrelated `profile_picture`). Don't guess it — log one real
`post.raw_data` from `fetchFacebookPosts` (`console.log(JSON.stringify(post.raw_data, null, 2))`
in `facebook/fetch.js`) and look for `full_picture`, `attachments.data[].media.image.src`,
or similar; Meta's Graph API conventionally uses one of those. Wire
whichever one actually appears the same way as the Instagram fix above.

## 4. Where to call the OCR service

Two viable hook points; recommend the first:

**A. Per-post, right before `upsertPost`** (in each platform's
`runProfile.js`) — OCR runs once per post the first time it's seen, and the
result is already attached by the time it's written to the DB:

```js
// runProfile.js, inside the existing loop
for (const post of posts) {
  if (!(await stillStarted(accountId, prisma))) break;
  await enrichPostWithOcr(post);          // <-- new; see §6 for the function
  const result = await upsertPost(post, { db: prisma, dbName });
  if (result.created) postsNew += 1;
  else postsUpdated += 1;
}
```

This only OCRs posts as they're first fetched or re-fetched — cheap, and
matches how `upsertPost` already only re-triggers sentiment analysis when
something changed.

**B. A separate backfill/worker job** reading `social_media_posts` where
`media_urls` is non-empty and no OCR result exists yet, processing older
rows independently of the live fetch loop. Better if OCR turns out too
slow to run inline with fetching (see §7's latency numbers) or if you want
retries decoupled from the fetch schedule. Not built here — worth it only
if (A) proves too slow in practice.

Start with (A); it's a few lines, not a new job/queue to operate.

## 5. Storing the result — no schema migration required to start

Two places to put it, both viable without touching `prisma/tenant.schema.prisma`:

1. **`raw_data.ocr`** — structured, doesn't disturb `text`:
   ```js
   post.raw_data = { ...post.raw_data, ocr: ocrResult };
   ```
2. **Appended into `text`** — the pragmatic win: `upsertPost` already
   re-triggers sentiment analysis (`enqueuePost`) whenever `text` changes,
   with zero other code changes. A meme/poster/notice image's actual
   content currently never reaches sentiment analysis at all (only the
   caption does) — appending gets it there for free:
   ```js
   if (ocrResult?.full_text) {
     post.text = [post.text, ocrResult.full_text].filter(Boolean).join('\n\n[Image text]\n');
   }
   ```

Recommend doing **both**: (1) for a clean structured record, (2) for the
existing analysis pipeline to actually see it. If OCR text quality turns
out to need filtering (low-confidence blocks, see `API.md`'s note on
`blocks[].confidence`) before it's trusted enough to feed sentiment
analysis, only do (1) at first and add (2) once that's judged reliable
enough for your data.

If OCR results need to be queryable/reportable as first-class fields later
(not just JSON-nested), that's when a real migration
(`ocr_text String? @db.Text`, `ocr_processed_at DateTime?`) is worth it —
not required to ship the first version.

## 6. The actual integration code

`enrichPostWithOcr`, matching this codebase's own conventions (CommonJS,
async/await, the same error-swallowing-with-warn pattern `fetch.js` already
uses for reels failures):

```js
// e.g. backend/src/services/monitoringsocialmedia/ocrEnrich.js
const axios = require('axios');

const OCR_BASE_URL = process.env.OCR_SERVICE_URL || 'http://98.86.63.69:8000';
const OCR_TIMEOUT_MS = Number(process.env.OCR_TIMEOUT_MS || 60000);

/**
 * Downloads a post's first image server-side (reusing the same
 * Referer-aware fetch this codebase already uses in modules/media/media.routes.js
 * for cdninstagram.com/fbcdn.net/etc.) and sends the BYTES to the OCR
 * service as base64 -- not the raw CDN URL. Platform CDNs commonly reject
 * hotlinked fetches without the right Referer; media.routes.js's own
 * refererFor() function is proof this codebase already had to solve that
 * exact problem once. Sending bytes sidesteps it a second time.
 */
const enrichPostWithOcr = async (post) => {
  const imageUrl = Array.isArray(post.media_urls) ? post.media_urls[0] : null;
  if (!imageUrl) return; // nothing to OCR -- text-only post, or media_urls still empty (see §3)

  try {
    const imageResp = await axios.get(imageUrl, {
      responseType: 'arraybuffer',
      timeout: 30000,
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible; BluraSaga/1.0)' },
      // Add the same Referer header modules/media/media.routes.js's refererFor() would
      // pick for this host if this image URL turns out to need one too.
    });
    const imageBase64 = Buffer.from(imageResp.data).toString('base64');

    const { data: result } = await axios.post(
      `${OCR_BASE_URL}/extract`,
      { image_base64: imageBase64 },
      { timeout: OCR_TIMEOUT_MS }
    );

    if (!result.success) {
      console.warn(`[ocr] extraction failed for ${post.external_id}: ${result.error}`);
      return;
    }

    post.raw_data = { ...(post.raw_data || {}), ocr: result.data };
    if (result.data.full_text) {
      post.text = [post.text, result.data.full_text]
        .filter(Boolean)
        .join('\n\n[Image text]\n');
    }
  } catch (err) {
    // Never let an OCR failure block ingestion -- same philosophy as the
    // existing `try { reels } catch { console.warn }` in fetch.js.
    console.warn(`[ocr] skipped for ${post.external_id}: ${err.message}`);
  }
};

module.exports = { enrichPostWithOcr };
```

Then in `runProfile.js`:

```js
const { enrichPostWithOcr } = require('../ocrEnrich');
// ... inside the posts loop, before upsertPost — see §4
```

## 7. Operational things to get right before flipping this on

- **Security group.** `acb`'s AWS security group does not yet allow inbound
  traffic on port 8000 from anywhere (see `DEPLOYMENT.md` §5) — this
  integration will get connection-refused/timeout until that's opened for
  wherever saga-police actually runs. Not something either codebase's code
  can fix; it's an AWS console change.
- **No auth on `/extract` yet.** Fine while saga-police is the only caller
  and the security group is locked to its IP; revisit if that stops being
  true (`DEPLOYMENT.md` §7).
- **Latency: ~4-12 seconds per image, one at a time.** The OCR service
  processes one image at a time by design — measured faster than running
  multiple in parallel on its single GPU (`DEPLOYMENT.md` §8). If
  Instagram/Facebook accounts post images frequently across many tenants,
  calling this inline in the fetch loop (§4 option A) will serialize
  behind the OCR service's own queue. Set `OCR_TIMEOUT_MS` generously
  (60s+) and expect a fetch tick with N new images to take roughly
  `N × 4-12s` longer than it does today. If that's a problem in practice,
  switch to option B (a separate backfill worker) rather than tuning this
  further blind.
- **Cold-start / first request after a restart.** If the OCR service was
  just restarted, `/health` may report `model_loaded: true` before it's
  actually warmed up, or the first request may simply queue behind
  warmup — see `DEPLOYMENT.md`'s cold-start note. Not something to special-case
  in saga-police; the timeout above already covers it.
- **Port collision, dev machines only.** saga-police's own default `PORT`
  (`backend/src/index.js`) is also `8000`. Irrelevant once deployed
  separately (as it is now — saga-police here, OCR on `acb`), but avoid
  running both locally on the same machine without overriding one.
- **`OCR_SERVICE_URL` as an env var**, not a hardcoded IP — `acb`'s address
  or the service's port could change; keep it out of committed code the
  same way `PUBLIC_BACKEND_URL`/`MEDIA_DOWNLOAD_TIMEOUT_MS` etc. are
  already handled in `modules/media/media.routes.js`.

## 8. Verifying the wiring, before touching the scheduler

Test connectivity and the response shape in isolation first — this needs
no saga-police code, just confirms the two services can actually talk:

```bash
curl -X POST http://98.86.63.69:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"image_url": "<any real Instagram/Facebook/Telegram image URL you have handy>"}'
```

If that 400s/422s on a real CDN URL specifically (works fine on a plain
image host), that's the Referer-hotlinking issue §6 already designs
around — confirms you need the download-then-base64 path, not a reason to
debug the OCR service itself.

Then a minimal Node script exercising the real function before wiring it
into the scheduler:

```js
const { enrichPostWithOcr } = require('./src/services/monitoringsocialmedia/ocrEnrich');

(async () => {
  const post = { external_id: 'test', media_urls: ['<a real image URL>'], text: null, raw_data: {} };
  await enrichPostWithOcr(post);
  console.log(post.raw_data.ocr?.full_text);
})();
```

## Reference

- [API.md](API.md) — full `/extract` request/response contract, error codes, curl/Python/Node examples not specific to saga-police.
- [DEPLOYMENT.md](DEPLOYMENT.md) — how the OCR service itself is run, PM2, the concurrency measurements behind §7's latency numbers.
