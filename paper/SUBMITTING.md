# Submitting the Paper Track writeup

Read from the competition's own pages (Kaggle API, `get_hackathon_overview`, 26 October 2026).

* **Deadline:** 9 November 2026, 23:59 UTC on Kaggle (the ARC Prize site says 8 November;
  submit by the 8th). Drafts and unsubmitted writeups are not judged.
* **What is submitted:** a Kaggle Writeup, made in the browser ("New Writeup" on
  https://www.kaggle.com/competitions/arc-prize-2026-paper-track/projects); the API can read
  writeups but not create them. It needs:
  1. a title, a subtitle, and the text: at most 1,500 words, or it may be penalised
     (`paper/writeup.md`);
  2. a **cover image** in the Media Gallery, required (`paper/cover.png`), and a "Card and
     Thumbnail Image" of 560 × 280 (`paper/thumbnail.png`, from `paper/make_thumbnail.py`: the
     cover's small text does not read at that size);
  3. the **public notebook** in the Project Links field: `ahmedabdalatef/neo4-arc-agi-3` (a
     private notebook is made public after the deadline anyway);
  4. optionally a public link to a PDF version;
  5. the Track: "Main Track"; then "Submit" at the top right.
* **Accuracy** is scored from the leaderboard: the writeup is matched to the ARC-AGI-3
  submission by its ID. Submissions so far: 57659417 (v1, 0.33), 57687937 (v5, 0.36),
  57717349 (v7, 0.46), 57752153 (v8, 0.33), 57851197 (v9, 0.46), 57865735 (v10, 0.46).
* **The team** must be the one that submitted to ARC-AGI-3 (the account `ahmedabdalatef`).
* **Prizes:** 1st $50,000, 2nd $20,000, 3rd $5,000; up to $375,000 more, shared equally, for
  papers scoring above 4.5 of 5. Winners' papers are licensed CC-BY 4.0.
* **Rubric** (0–5 each, averaged): Accuracy, Universality, Progress, Theory, Completeness,
  Novelty.

## What to choose (decided 1 November 2026)

* **The writeup's submission: v10** — ref 57865735, notebook `ahmedabdalatef/neo4-arc-agi-3`
  version 10, bundle of commit `130f619`. It is the code the paper describes, through Phase 72
  (first return, then explore). Put the notebook in the Project Links field.
* **The two final submissions for ARC-AGI-3: v10 (57865735) and v7 (57717349).** v7, v9 and v10
  tie at 0.46 on the public leaderboard; the final ranking is on the private score, which is not
  shown. v10 is the best on the holdout (0.70 against 0.62). v7 is the most different from it
  (without the record that judges goals, Phase 70, or the return to the best state, Phase 72),
  so it covers the case where those two do worse on the private games. v9 lies between the two
  and would add little. The API cannot select final submissions: on the competition's
  "Submissions" page, tick "Use for Final Score" for these two. Left unticked, Kaggle chooses by
  public score, and the three tie. **Done: v7 and v10 ticked on 1 November.**

## Before pressing Submit (checked 1 November 2026)

* **Make the repository public.** The paper links its code at https://github.com/ahmedmah987/neo4,
  which is private: a reader gets a 404. GitHub: Settings → General → Danger Zone → Change
  visibility → Public. No secrets, tokens or credentials files are tracked (99 files checked by
  name), and nothing from other private projects.
* **Title:** Search is not the bottleneck: what a frames-only ARC-AGI-3 agent pays for (73
  characters: Kaggle cuts titles at 80, and cut the first one, which ended ", and why", to
  "…pays for, and w")
* **Subtitle:** A language-model-free agent, and why a level learned in L lives is worth 1/L² of
  its score
* **Text:** `paper/writeup.md` without its title line (the title goes in its own field): 1,475
  words counting only words with a letter or digit, 1,494 counting every token of the rendered
  text, symbols included.
* **References:** the five arXiv numbers resolve to the papers named (checked against arXiv);
  Twin's sentence is quoted as its abstract has it.

## Submitted (2 November 2026)

* **The writeup:** https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/search-is-not-the-bottleneck-what-a-frames-only-a
  (hackathon writeup 91362, writeup 123019), published 2 November 00:44 UTC, Main Track, team
  AHMED ABDALATEF, licence CC BY 4.0. A visitor not signed in gets a 404 until Kaggle opens the
  writeups. Read back through the API (`GetHackathonWriteUp`, and `ListHackathonWriteUps` with
  `published` set; without it the call is refused).
* **Fixed, and read back through the API at 00:57 UTC:** the title (cut at 80 characters
  first) is the 73-character one above; the text is `paper/writeup.md` without its title line,
  identical to it, 1,475 words; Project Links hold the notebook
  `https://www.kaggle.com/code/ahmedabdalatef/neo4-arc-agi-3` (version 10, still private) and the
  GitHub repository; the gallery holds one image, the cover. A paste from rendered text drops the
  Markdown (headings, tables, bold, lists): paste the raw file, e.g. from raw.githubusercontent.com.
* **Not checkable from outside:** the gallery cover, the card and the thumbnail are served only
  to a signed-in viewer; check in the browser that the gallery shows the 16:9 `cover.png` and the
  thumbnail the blue square.
* **ARC-AGI-3** closed on 2 November 23:59 UTC with v7 and v10 selected; the private scores are
  posted once the organisers verify the final leaderboard. On 4 November the Paper Track's
  Writeups tab, seen signed out, still listed no writeup, as expected before the deadline.
* **8 November:** no private score yet (the API's `hasPrivateScore` is false for all six
  submissions; the leaderboard page: "The private leaderboard will become available after the
  results are verified"; winners are announced on 4 December). On the public leaderboard
  (downloaded 8 November 01:42 UTC) the team ranks 2,966th of 4,926 with 0.46: 2,232 teams
  score above 20 and 973 above 30, only 28 above 40 and 3 above 60 (top 75.56).
