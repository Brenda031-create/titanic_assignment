import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def prepare_features(df):
    """
    Prepares dataset for feature selection.
    """

    df = df.copy()

    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())

    if 'Embarked' in df.columns and not df['Embarked'].mode().empty:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    if {'SibSp', 'Parch'}.issubset(df.columns):
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Drop non-useful columns
    drop_cols = ['Name', 'Ticket', 'PassengerId', 'Cabin']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    categorical_cols = [
        col for col in df.select_dtypes(include=['object', 'category']).columns
        if col != 'Survived'
    ]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    return df


def get_feature_importance(df):
    """
    Trains a Random Forest model and returns feature importance.
    """

    df = prepare_features(df)

    if 'Survived' not in df.columns:
        raise KeyError("Input dataframe must contain a 'Survived' column.")

    # Separate features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Get importance
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)

    return importance


def select_top_features(df, top_n=10):
    """
    Selects top N important features.
    """

    df = prepare_features(df)
    importance = get_feature_importance(df)
    top_features = importance.head(top_n).index.tolist()

    return df[top_features + ['Survived']]