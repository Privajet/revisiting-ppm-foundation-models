from sklearn.metrics import accuracy_score, mean_squared_error
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.many_class import ManyClassClassifier

TABPFN3_CLS_CKPT = "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"
TABPFN3_REG_CKPT = "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"

def run_tabpfn3_baseline(train_log, test_log, random_state: int):
    """
    - Classification: next_activity
    - Regression: remaining_time, time_to_next_event
    """

    max_n_train = 50000
    max_n_test = 2000
    
    df_train = train_log.dataframe
    df_test = test_log.dataframe
    
    if len(df_train) > max_n_train:
        df_train = df_train.sample(n=max_n_train, random_state=random_state).reset_index(drop=True)
        
    if len(df_test) > max_n_test:
        df_test = df_test.sample(n=max_n_test, random_state=random_state).reset_index(drop=True)

    feature_cols = list(train_log.features.categorical) + list(train_log.features.numerical)

    X_train = df_train[feature_cols].to_numpy()
    X_test = df_test[feature_cols].to_numpy()

    metrics = {}

    # Classification: next_activity
    y_train_cls = df_train["next_activity"].to_numpy()
    y_test_cls = df_test["next_activity"].to_numpy()

    base_clf = TabPFNClassifier(model_path=TABPFN3_CLS_CKPT)
    clf = ManyClassClassifier(
        estimator=base_clf,
        alphabet_size=10,         
        n_estimators=None,           
        n_estimators_redundancy=4,   
        random_state=random_state,
        verbose=0,
        codebook_config=None,
        row_weighting_config=None,
        aggregation_config=None,
    )
    clf.fit(X_train, y_train_cls)
    y_pred_cls = clf.predict(X_test)

    metrics["test_next_activity_acc"] = float(accuracy_score(y_test_cls, y_pred_cls))

    # Regression: remaining_time 
    y_train_rt = df_train["remaining_time"].astype(float).to_numpy()
    y_test_rt = df_test["remaining_time"].astype(float).to_numpy()

    reg_rt = TabPFNRegressor(model_path=TABPFN3_REG_CKPT)
    reg_rt.fit(X_train, y_train_rt)
    pred_rt = reg_rt.predict(X_test)

    metrics["test_next_remaining_time_loss"] = float(mean_squared_error(y_test_rt, pred_rt))

    # Regression: time_to_next_event 
    y_train_nt = df_train["time_to_next_event"].astype(float).to_numpy()
    y_test_nt = df_test["time_to_next_event"].astype(float).to_numpy()

    reg_nt = TabPFNRegressor(model_path=TABPFN3_REG_CKPT)
    reg_nt.fit(X_train, y_train_nt)
    pred_nt = reg_nt.predict(X_test)

    metrics["test_next_time_to_next_event_loss"] = float(mean_squared_error(y_test_nt, pred_nt))
    
    metrics["y_true_pred_dump"] = {
        "y_true_next_activity": y_test_cls,
        "y_pred_next_activity": y_pred_cls,
        "y_true_remaining_time": y_test_rt,
        "y_pred_remaining_time": pred_rt,
        "y_true_time_to_next_event": y_test_nt,
        "y_pred_time_to_next_event": pred_nt,
    }
    
    return metrics