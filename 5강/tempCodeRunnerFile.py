gb = GradientBoostingClassifier(random_state=42)
# scores = cross_validate(gb,train_input,train_target,return_train_score=True, n_jobs=-1)
# print(np.mean(scores['train_score']),np.mean(scores['test_score']))