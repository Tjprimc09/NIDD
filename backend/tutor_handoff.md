# NIDD Backend - Tutor Handoff Notes

**To the next AI Tutor:**
The user is working on `backend/train.py`, continuing Task 2 of the Network Intrusion Detection Dashboard.
The user has specifically requested you walk them through the code **ONE STEP AT A TIME**. Socratic method only. Do NOT overwhelm them with long checklists, and do NOT write out completed code blocks for them.

**Current Progress:**
- The data is loaded, cleaned, and mapped to 5 broad attack categories.
- The `ColumnTransformer` (preprocessor) is built for numeric/categorical data.
- The baseline models `LogReg`, `RandForest`, and `MLP` were successfully trained in a loop.
- The user has created a cleanly trimmed `param_grid` dictionary to tune the `MLPClassifier` using `GridSearchCV`.
- The user accurately added `from sklearn.model_selection import GridSearchCV` at the top of their script.

**Next Immediate Step:**
- Pick up right exactly here. The user needs to build the single dedicated `mlp_pipeline` (putting the preprocessor and the `MLPClassifier(random_state=42)` together into a Pipeline) right below their `param_grid`. 
- **DO NOT write the code for them.** Validate their setup and ask them a guiding question on how they might build that dedicated pipeline object for the MLP.
