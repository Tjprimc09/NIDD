# Pandas Cheat Sheet for ML Projects

## Loading Data
| Method | What It Does | Example |
|---|---|---|
| `pd.read_csv(path)` | Load CSV/TXT into DataFrame | `pd.read_csv("data.txt", header=None, names=cols)` |

## Exploring Data
| Method | What It Does | Example |
|---|---|---|
| `df.head(n)` | First n rows | `df.head(10)` |
| `df.tail(n)` | Last n rows | `df.tail(5)` |
| `df.shape` | (rows, columns) count | `df.shape` → `(125973, 43)` |
| `df.info()` | Column types + non-null counts | `df.info()` |
| `df.describe()` | Stats (mean, std, min, max) | `df.describe()` |
| `df.columns` | List all column names | `df.columns` |
| `df.dtypes` | Data type of each column | `df.dtypes` |

## Selecting Data
| Method | What It Does | Example |
|---|---|---|
| `df['col']` | Get one column | `df['label']` |
| `df[['a','b']]` | Get multiple columns | `df[['age','bmi']]` |
| `df.loc[rows, cols]` | Select by label/condition | `df.loc[df['label']=='dos', 'label']` |
| `df.iloc[rows, cols]` | Select by index position | `df.iloc[0:10, 0:5]` |

## Unique Values & Counting
| Method | What It Does | Example |
|---|---|---|
| `df['col'].unique()` | Array of unique values | `df['label'].unique()` |
| `df['col'].nunique()` | Count of unique values | `df['label'].nunique()` → `23` |
| `df['col'].value_counts()` | Count per unique value | `df['label'].value_counts()` |

## Modifying Data
| Method | What It Does | Example |
|---|---|---|
| `df.drop('col', axis=1)` | Remove a column | `df.drop('difficulty', axis=1, inplace=True)` |
| `df.drop(0, axis=0)` | Remove a row | `df.drop(0, axis=0)` |
| `df['col'].replace(dict)` | Replace values using a mapping | `df['label'].replace({'neptune':'dos'})` |
| `df['col'].map(func)` | Apply a function to each value | `df['label'].map(my_function)` |
| `df['col'].apply(func)` | Apply function (more flexible) | `df['col'].apply(lambda x: x*2)` |
| `df.rename(columns=dict)` | Rename columns | `df.rename(columns={'old':'new'})` |
| `df['new'] = value` | Add a new column | `df['category'] = 'unknown'` |

## Filtering
| Method | What It Does | Example |
|---|---|---|
| `df[df['col'] > val]` | Filter rows by condition | `df[df['duration'] > 0]` |
| `df[df['col'].isin(list)]` | Filter by membership | `df[df['label'].isin(['dos','probe'])]` |
| `df.query('expression')` | SQL-like filtering | `df.query('duration > 0 and land == 1')` |

## Sorting
| Method | What It Does | Example |
|---|---|---|
| `df.sort_values('col')` | Sort by column | `df.sort_values('duration', ascending=False)` |

## Missing Data
| Method | What It Does | Example |
|---|---|---|
| `df.isnull().sum()` | Count missing per column | `df.isnull().sum()` |
| `df['col'].fillna(val)` | Fill missing with a value | `df['col'].fillna(df['col'].mean())` |
| `df.dropna()` | Remove rows with missing data | `df.dropna()` |

## Grouping
| Method | What It Does | Example |
|---|---|---|
| `df.groupby('col').mean()` | Group + average | `df.groupby('label').mean()` |
| `df.groupby('col').size()` | Group + count | `df.groupby('label').size()` |
| `df.groupby('col').agg(dict)` | Group + multiple stats | `df.groupby('label').agg({'duration':'mean'})` |

## Key Concepts
- **`axis=0`** → rows (vertical ↕)
- **`axis=1`** → columns (horizontal ↔)
- **`inplace=True`** → modify original, no reassignment needed
- **`inplace=False`** (default) → returns a copy, must reassign
