# Manual Semantic Audit — Samples and Verdicts (L4)

Samples drawn with `random.seed(42)`, stratified 7 items per factory, by the sampling code in
`assess.py` / the audit-sample block documented below. Precision sample: 42 events, each judged
against all transcript entries overlapping the event interval. Recall sample: 42 transcript
entries, each classified operational/non-operational and checked for overlap with ≥1 event.

Verdict codes — precision: `C` fully correct (label faithfully describes transcribed behavior in
the interval), `P` partially supported (plausible, consistent with surrounding context, but not
verifiable from the overlapping entry alone — e.g., "the repetitive task continues"), `X`
contradicted (label conflicts with transcript). Recall: `COV` covered by ≥1 event, `EXC` uncovered
and non-operational (correct exclusion), `MISS` uncovered but operational (extraction miss).

## Precision sample (n = 42): 36 C, 6 P, 0 X → strict precision 0.857

| # | Factory/Worker/Video | Event interval | Activity label | Verdict | Note |
|---|---|---|---|---|---|
| 1 | 001/w003/v00009 | 00:01:38–00:02:12 | mate components over central jig | C | assembly over metal block |
| 2 | 001/w002/v00001 | 00:08:41–00:08:48 | drop sealed bags into bulk container | C | |
| 3 | 001/w006/v00000 | 00:07:50–00:08:02 | receive part from adjacent coworker | C | |
| 4 | 001/w005/v00006 | 00:17:09–00:17:15 | actuate machine press | P | interval covers full load–press–remove cycle |
| 5 | 001/w005/v00004 | 00:14:00–00:14:55 | place main component into larger bag | P | transcript: "repetitive assembly and packaging continues" |
| 6 | 001/w004/v00000 | 00:06:34–00:06:43 | drive fastener into metal parts | P | transcript: "repetitive assembly continues" |
| 7 | 001/w003/v00006 | 00:04:15–00:06:22 | insert small pin into base components | C | split of long composite entry |
| 8 | 002/w009/v00000 | 00:19:07–00:20:00 | align printed fabric patterns | C | |
| 9 | 002/w006/v00002 | 00:05:39–00:06:11 | enclose garment in pre-printed cardboard folder | C | |
| 10 | 002/w002/v00007 | 00:10:06–00:10:18 | attach black plastic hanger hook to bag | C | |
| 11 | 002/w007/v00002 | 00:08:16–00:08:31 | wrap white cardboard band around folded shirt | C | |
| 12 | 002/w005/v00007 | 00:06:09–00:06:24 | fold t-shirt over removable white template | C | split of repeated-cycle entry |
| 13 | 002/w002/v00001 | 00:12:22–00:12:26 | drop packaged garments into bulk box | P | transcript: "appearing to drop an item" |
| 14 | 002/w002/v00001 | 00:11:41–00:11:43 | drop packaged garments into bulk box | C | |
| 15 | 003/w003/v00000 | 00:10:36–00:13:22 | scrape component surface with metal tool | C | |
| 16 | 003/w005/v00004 | 00:04:13–00:04:29 | strike metal casting with hammer | C | |
| 17 | 003/w005/v00004 | 00:15:47–00:15:53 | scoop loose parts from bulk bin | C | |
| 18 | 003/w010/v00000 | 00:11:45–00:12:05 | squeeze binding paste from piping bag | C | |
| 19 | 003/w001/v00001 | 00:05:45–00:06:12 | fill metal gap with dark clay | C | |
| 20 | 003/w010/v00003 | 00:17:12–00:17:29 | load pre-manufactured insert into mold | C | |
| 21 | 003/w005/v00000 | 00:03:59–00:07:18 | strike metal casting with hammer | C | |
| 22 | 004/w007/v00001 | 00:14:07–00:14:36 | Iron folded garment with steam iron | C | |
| 23 | 004/w006/v00004 | 00:16:20–00:16:29 | Unfold sewn piece to inspect seam tension | C | |
| 24 | 004/w007/v00001 | 00:07:22–00:07:24 | Snap large cardboard sheet in half | C | |
| 25 | 004/w006/v00002 | 00:17:00–00:17:03 | Snip loose threads with small clippers | C | |
| 26 | 004/w004/v00005 | 00:14:08–00:14:16 | Push finished fabric chain to table edge | C | |
| 27 | 004/w003/v00000 | 00:00:00–00:01:00 | Hem large fabric piece in short bursts | C | merges 5 s stand-up interruption |
| 28 | 004/w005/v00001 | 00:08:44–00:15:18 | Sew short straight seam | C | aggregated repetition |
| 29 | 005/w004/v00002 | 00:04:43–00:04:51 | brush metal shavings off lathe tool post | C | |
| 30 | 005/w001/v00000 | 00:06:26–00:06:36 | position metallic nameplate on motor housing | C | split of ten-motor loop entry |
| 31 | 005/w003/v00000 | 00:08:20–00:08:27 | advance cutting tool via carriage handwheels | C | |
| 32 | 005/w008/v00001 | 00:19:10–00:19:18 | arrange parts manually inside deep wire mesh bin | P | tail of composite transport entry |
| 33 | 005/w006/v00000 | 00:06:03–00:06:10 | initiate milling machine cycle via control panel | C | |
| 34 | 005/w003/v00000 | 00:07:22–00:07:29 | advance cutting tool via carriage handwheels | C | |
| 35 | 005/w004/v00000 | 00:02:23–00:02:33 | apply adhesive label to motor housing | C | |
| 36 | 006/w003/v00002 | 00:12:28–00:12:38 | lift empty mold box onto pattern plate | C | "onto the machine" |
| 37 | 006/w001/v00004 | 00:05:08–00:06:01 | sift through mixed small parts on floor | C | floor bin |
| 38 | 006/w001/v00002 | 00:08:42–00:08:58 | guide upper mold half onto lower mold half | C | |
| 39 | 006/w003/v00003 | 00:11:39–00:11:55 | pack casting sand into mold frame | C | |
| 40 | 006/w001/v00002 | 00:11:00–00:11:07 | guide upper mold half onto lower mold half | P | transcript vague: "helps with a mold" |
| 41 | 006/w003/v00003 | 00:04:24–00:04:36 | level powder surface with wooden scraper | C | |
| 42 | 006/w003/v00002 | 00:16:15–00:16:30 | pack casting sand into mold frame | C | |

## Recall sample (n = 42): 34 COV, 4 EXC, 4 MISS → operational recall 34/38 ≈ 0.895

| # | Factory/Worker/Video | Entry interval | Verdict | Content (abridged) |
|---|---|---|---|---|
| 1 | 001/w008/v00010 | 00:12:00–00:12:51 | COV | continues repetitive assembly |
| 2 | 001/w004/v00007 | 00:05:22–00:05:39 | COV | tears open bag of new metal parts |
| 3 | 001/w002/v00003 | 00:01:07–00:01:12 | COV | staples bag shut, sets aside |
| 4 | 001/w010/v00002 | 00:11:22–00:12:05 | COV | aligns hinged-clip rail on plate in jig |
| 5 | 001/w007/v00004 | 00:15:43–00:16:41 | COV | resumes stamping operation |
| 6 | 001/w008/v00002 | 00:08:46–00:08:49 | COV | camera tilt (overlapped by longer event) |
| 7 | 001/w003/v00005 | 00:09:10–00:09:15 | COV | places assembled parts onto cardboard |
| 8 | 002/w005/v00009 | 00:20:10–00:20:13 | COV | places cardboard piece on t-shirt |
| 9 | 002/w003/v00000 | 00:09:16–00:09:44 | COV | fetches and aligns garment stacks |
| 10 | 002/w010/v00001 | 11:51:00–12:59:00 | MISS | aligning stack of pink fabric pieces (multi-hour recording; truncation regime) |
| 11 | 002/w005/v00003 | 00:03:38–00:03:47 | COV | irons folded fabric |
| 12 | 002/w005/v00009 | 00:09:44–00:09:49 | COV | irons t-shirt back |
| 13 | 002/w004/v00006 | 00:13:17–00:13:20 | EXC | short walking sequence |
| 14 | 002/w002/v00009 | 00:11:01–00:12:35 | COV | two workers fill sack with garments |
| 15 | 003/w001/v00002 | 00:18:12–00:18:42 | COV | continues scraping task |
| 16 | 003/w009/v00003 | 00:19:35–00:19:42 | COV | empties box onto tray |
| 17 | 003/w004/v00002 | 00:02:58–00:03:08 | COV | applies white paste with brush |
| 18 | 003/w004/v00007 | 00:19:45–00:19:55 | COV | continues filing metal part |
| 19 | 003/w002/v00000 | 00:18:30–00:18:50 | MISS | fetches empty tray, stages tied bundles |
| 20 | 003/w004/v00002 | 00:06:27–00:06:33 | COV | brushes paste onto part |
| 21 | 003/w002/v00004 | 00:02:48–00:02:51 | COV | drops debris into box |
| 22 | 004/w006/v00004 | 00:17:28–00:17:34 | COV | checks seam, folds fabric |
| 23 | 004/w005/v00003 | 00:00:40–00:00:48 | COV | prepares fabric edge for sewing |
| 24 | 004/w009/v00001 | 00:09:44–00:09:46 | COV | drops packaged stack in box |
| 25 | 004/w006/v00004 | 00:08:12–00:08:29 | COV | aligns fabric, resumes sewing |
| 26 | 004/w003/v00000 | 00:19:50–00:20:00 | COV | resumes continuous sewing |
| 27 | 004/w006/v00004 | 00:11:38–00:11:43 | EXC | checks phone briefly |
| 28 | 004/w006/v00004 | 00:01:24–00:01:40 | COV | sews another section |
| 29 | 005/w002/v00000 | 00:08:04–00:08:33 | EXC | brief walk around workbench, returns |
| 30 | 005/w008/v00000 | 00:01:35–00:01:42 | EXC | camera shake while walking |
| 31 | 005/w003/v00000 | 00:05:32–00:05:40 | COV | tightens fixture with wrench |
| 32 | 005/w008/v00000 | 00:11:36–00:11:45 | MISS | standing, inspecting part (borderline operational) |
| 33 | 005/w011/v00000 | 00:08:56–00:09:22 | MISS | presses control-panel buttons, consults sheet |
| 34 | 005/w008/v00000 | 00:05:32–00:05:54 | COV | lifts bin with pallet jack, transports |
| 35 | 005/w006/v00001 | 00:02:00–00:04:00 | COV | workers stack red components into wire basket |
| 36 | 006/w001/v00002 | 00:08:58–00:09:12 | COV | waits with spray gun, sprays mold |
| 37 | 006/w007/v00000 | 00:03:13–00:03:22 | COV | hammers casting apart |
| 38 | 006/w007/v00000 | 00:15:10–00:15:23 | COV | hammers casting, scrapes sand |
| 39 | 006/w001/v00007 | 00:11:57–00:12:03 | COV | blows air onto pattern plate |
| 40 | 006/w005/v00001 | 00:02:47–00:02:55 | COV | carries mold, operates control panel |
| 41 | 006/w001/v00008 | 00:19:19–00:19:32 | COV | flips pattern plate, air-cleans |
| 42 | 006/w001/v00007 | 00:09:24–00:09:35 | COV | two workers move sand mold with lifting clamp |
