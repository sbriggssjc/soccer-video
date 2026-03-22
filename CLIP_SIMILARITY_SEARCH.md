# Clip Similarity Search

## Task
- Find a highlight clip of a professional soccer player that looks substantially similar to the user-supplied clip `NEOFC - 27 (Claire).mp4`.

## Objective
- Identify the action pattern, camera feel, and player movement in the source clip.
- Recommend one or more professional clips that are visually and tactically similar.

## Current Understanding
- Source clip path: `C:\Users\scott\OneDrive\Desktop\Uploads\Anything You Can Do - Clip\NEOFC - 27 (Claire).mp4`
- Clip duration: about 6.07 seconds.
- Export format: vertical social crop, `1080x1920`, 30 fps.
- Initial visual read: wide sideline view of a winger/attacker in a shoulder-to-shoulder duel, beating pressure on the flank and carrying the ball forward into open space.

## What Worked
- `ffprobe` returned clean metadata for the source clip.
- `ffmpeg` successfully generated a contact sheet for fast visual inspection.

## What Did Not Work
- The contact sheet is useful for broad identification, but it does not fully capture the movement details, so additional frame inspection may still be needed.

## Search Direction
- Prioritize professional women's soccer first because the source clip appears to be from a girls/women's match.
- Look for clips with these shared traits:
  - wide touchline camera angle
  - isolated 1v1 on the wing
  - attacker rides or evades contact
  - burst into space after the duel

## Alignment
- Matching on movement pattern and shot composition should produce a closer result than matching on outcome alone.

## Findings
- The source clip reads most like a solo carry through central-right traffic rather than a pure touchline dribble.
- Best professional similarity match: Sophia Smith vs. Kansas City Current on June 23, 2024.
- Strong alternate: Trinity Rodman vs. OL Reign on March 26, 2023.
- Style alternate: Salma Paralluelo vs. Netherlands on August 11, 2023.

## Recommendation Notes
- Sophia Smith is the closest match on movement profile: receives under pressure, turns through defenders, powers forward with close control, then finishes.
- Trinity Rodman is the next-best match if the user wants a longer open-field acceleration carry.
- Salma Paralluelo is useful if the user wants a World Cup-level version of a pace-driven break into the box.

## Refined Target
- Prioritize near-matches on these specifics:
  - elevated sideline camera from the attacking half
  - attacker carries from the right channel toward goal
  - defender pressure is active through the run, not a clean breakaway
  - finish is driven low across goal into the far side
- Mens and womens clips are both acceptable if the visual and tactical resemblance is stronger.

## Refined Findings
- Exact matches on all dimensions are hard to find because pro footage usually uses broadcast angles that are less similar to the elevated touchline phone angle in the source clip.
- Best camera-angle analogs are often school/youth clips from MaxPreps or similar elevated sideline uploads.
- Best action-and-finish analogs are usually professional broadcast clips.
