# CLAUDE.md

## Project

This repository contains a hackathon MVP for AI-powered disaster damage assessment and spatial response prioritization using satellite imagery.

The complete specification is located at:

`doc/project-spec`

Read the complete specification before making architectural or implementation decisions.

## Core Requirements

- Build a working MVP, not a UI-only prototype.
- Actual AI inference is mandatory.
- Use the specified pretrained SegFormer-B0 xBD checkpoint.
- Do not train or fine-tune a model.
- The post-disaster image is the AI model input.
- The pre-disaster image is for visual comparison only.
- Never claim that the selected model performs before/after change detection.
- Never substitute xBD ground-truth labels for AI predictions.
- Never generate fake or hardcoded predictions in the production workflow.
- Do not describe pixel counts as building counts.
- Do not invent geographic coordinates.
- Keep the architecture lightweight and appropriate for a hackathon MVP.

## Technology Constraints

Use:

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Pillow / NumPy
- Pandas
- Plotly
- OpenCV only where useful

Do not introduce React, Next.js, FastAPI, databases, microservices, or unnecessary frameworks unless explicitly requested.

## Development Approach

Implement incrementally:

1. Get real SegFormer inference working.
2. Verify the segmentation output.
3. Implement and test the zone-ranking calculations.
4. Build the Streamlit dashboard.
5. Integrate all components.
6. Test end-to-end using real xBD sample imagery.

Do not build large portions of the UI around mocked AI output.

## Engineering Rules

- Prefer simple, readable implementations over abstractions.
- Do not over-engineer.
- Inspect existing code before changing it.
- Keep model loading cached.
- Support CPU and CUDA where available.
- Handle external model/download failures clearly.
- Keep scoring formulas transparent.
- Add tests for ranking calculations.
- Do not hard-code outputs simply to satisfy tests.

## Git

Work on feature branches rather than directly modifying `main`.

Before finishing a substantial task:

- run the relevant tests,
- check for obvious runtime errors,
- summarize what was implemented,
- report anything that could not be verified.

Do not force-push, delete branches, or perform destructive Git operations without approval.
