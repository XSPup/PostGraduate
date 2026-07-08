---
name: learning-project-record
description: Create consistent learning records for machine learning, deep learning, Python, GitHub, or coding practice projects. Use when the learner finishes or reviews a project and wants a structured report containing the project link, local path, learning goals, setup steps, key code concepts, problems encountered, results, and next actions.
---

# Learning Project Record

## Purpose

Use this skill to create or update one learning report per hands-on project. Keep the report beginner-friendly, concrete, and tied to what the learner actually did.

## Storage Rules

- Save reports under `D:\MyDeepLearning\learning-records\reports`.
- Name reports with `YYYY-MM-DD-short-project-name.md`.
- Keep one project per report.
- Include the GitHub URL, local path, environment, main files, learning goals, key concepts, errors, fixes, results, and next steps.
- Prefer Chinese report content when the learner is studying foundational ML concepts in Chinese.

## Report Format

Use this structure:

```markdown
# Project Learning Report: <Project Name>

## 1. Basic Information

- Date:
- Project name:
- GitHub link:
- Local path:
- Project type:
- Difficulty:

## 2. Why I Studied This Project

Explain the motivation and the knowledge points it supports.

## 3. Learning Goals

- Goal 1
- Goal 2
- Goal 3

## 4. What I Actually Did

Record the key actions in chronological order.

## 5. Project Structure

List important files and explain their roles.

## 6. Core Code Understanding

Record the code, formulas, or concepts that were actually understood.

## 7. Problems and Fixes

Use the pattern: problem -> cause -> fix -> what was learned.

## 8. Run Results

Record whether it ran, what figures appeared, and important output values.

## 9. Current Understanding

Summarize the algorithm or engineering workflow in the learner's own words.

## 10. Next Steps

List 2 to 5 concrete next actions.
```

## Quality Bar

- Do not write empty ceremonial sections.
- Do not overstate mastery; distinguish "can operate it" from "understands the principle".
- Preserve useful command lines exactly, especially setup commands and error fixes.
- When a bug was fixed, include the wrong command or code and the corrected version.
- Keep reports useful for the learner to re-read one month later.
