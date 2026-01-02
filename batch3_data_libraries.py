"""
Batch 3: Data Libraries Questions
- Pandas (15 questions)
- NumPy (15 questions)
"""

from database_manager import QuestionDatabase, create_question

def populate_pandas():
    """15 Pandas Questions"""
    questions = [
        create_question(
            "You have a DataFrame with missing values. What is the difference between df.dropna() and df.fillna(0)?",
            [
                "They do the same thing",
                "dropna() removes rows/columns with NaN; fillna(0) replaces NaN with 0",
                "dropna() replaces with 0; fillna() removes rows",
                "Both remove all data"
            ],
            1,
            "dropna() removes rows (axis=0, default) or columns (axis=1) containing NaN values. fillna(0) replaces all NaN values with 0 (or other specified value). dropna() reduces data size, fillna() preserves it. Use dropna() when missing data is minimal, fillna() when you want to impute. fillna() can use method='ffill' (forward fill) or method='bfill' (backward fill) for time series.",
            "Medium",
            85
        ),
        create_question(
            "What does df.groupby('category')['value'].mean() do?",
            [
                "Groups all data together",
                "Groups rows by 'category', then calculates mean of 'value' for each group",
                "Calculates mean of 'category'",
                "Removes the 'category' column"
            ],
            1,
            "groupby() splits data into groups based on 'category' values, then applies mean() to the 'value' column for each group, returning a Series indexed by category with corresponding means. This is split-apply-combine pattern. Can apply multiple aggregations: .agg(['mean', 'sum', 'count']). For multiple columns: groupby(['cat1', 'cat2']).",
            "Medium",
            80
        ),
        create_question(
            "In Pandas, what is the difference between df.loc[] and df.iloc[]?",
            [
                "They are identical",
                "loc uses labels (index/column names); iloc uses integer positions",
                "loc is for rows only",
                "iloc is deprecated"
            ],
            1,
            "loc[] uses label-based indexing: df.loc['row_name', 'col_name'] or df.loc[0:5] (includes end). iloc[] uses integer position-based indexing: df.iloc[0, 1] or df.iloc[0:5] (excludes end). Use loc for label-based access, iloc for position-based. loc is inclusive of end in slicing, iloc is not. Both support boolean indexing and can select rows and columns.",
            "Hard",
            95
        ),
        create_question(
            "What does pd.merge(df1, df2, on='key', how='left') do?",
            [
                "Combines dataframes vertically",
                "Performs a left join: keeps all rows from df1, matching rows from df2, NaN for non-matches",
                "Keeps only matching rows",
                "Removes the 'key' column"
            ],
            1,
            "Left join keeps all rows from the left DataFrame (df1) and matching rows from df2. Non-matching rows from df1 get NaN for df2 columns. how='inner' keeps only matches, how='outer' keeps all from both (with NaN for non-matches), how='right' keeps all from df2. Similar to SQL joins. on='key' specifies join column(s).",
            "Medium",
            90
        ),
        create_question(
            "You need to convert a 'date' column from string to datetime. What's the best approach?",
            [
                "df['date'] = int(df['date'])",
                "df['date'] = pd.to_datetime(df['date'])",
                "df['date'] = str(df['date'])",
                "df['date'].astype(datetime)"
            ],
            1,
            "pd.to_datetime() intelligently parses various date string formats and converts to datetime64 dtype. Usage: df['date'] = pd.to_datetime(df['date']). Can specify format for speed: format='%Y-%m-%d'. Handles errors with errors='coerce' (NaT for invalid) or errors='raise'. Once datetime, can use .dt accessor: df['date'].dt.year, .dt.month, etc.",
            "Medium",
            80
        ),
        create_question(
            "What does df.apply(lambda x: x.max() - x.min(), axis=0) do?",
            [
                "Finds max and min of entire DataFrame",
                "Applies function to each column (axis=0), returning the range (max-min) per column",
                "Applies to each row",
                "Removes outliers"
            ],
            1,
            "apply() applies function along specified axis. axis=0 (default) applies to each column (function receives column as Series). axis=1 applies to each row. Here, lambda receives each column and returns max-min (range). Result is a Series with range for each column. apply() is flexible but slower than vectorized operations. Use built-in methods when possible (e.g., df.max() - df.min()).",
            "Hard",
            95
        ),
        create_question(
            "What is the purpose of pd.concat([df1, df2], axis=0)?",
            [
                "Joins DataFrames on common columns",
                "Stacks DataFrames vertically (row-wise concatenation)",
                "Merges based on index",
                "Creates a copy of df1"
            ],
            1,
            "concat() with axis=0 (default) stacks DataFrames vertically (appends rows). axis=1 stacks horizontally (appends columns). ignore_index=True resets index. For vertical concat, columns must align; misaligned columns create NaN. Different from merge (which joins on keys) and join (index-based merge). Use concat for simple stacking, merge for key-based joins.",
            "Medium",
            80
        ),
        create_question(
            "What does df.pivot_table(values='sales', index='product', columns='region', aggfunc='sum') do?",
            [
                "Removes duplicates",
                "Creates a spreadsheet-style pivot table: products as rows, regions as columns, summed sales as values",
                "Transposes the DataFrame",
                "Filters data"
            ],
            1,
            "pivot_table() reshapes data: index becomes row labels, columns becomes column labels, values are aggregated using aggfunc. Here: products × regions table with summed sales. Handles duplicates via aggregation (unlike pivot()). aggfunc can be 'mean', 'sum', 'count', or custom function. fill_value=0 replaces NaN. Powerful for creating summary tables and reports.",
            "Hard",
            95
        ),
        create_question(
            "In Pandas, what is the difference between df.copy() and df.copy(deep=True)?",
            [
                "No difference, deep=True is default",
                "copy() creates a shallow copy; copy(deep=True) creates a deep copy of data and indices",
                "deep=True compresses the data",
                "copy() is faster"
            ],
            0,
            "Actually, deep=True is the default! df.copy() and df.copy(deep=True) both create deep copies (data and indices are copied). deep=False creates a shallow copy (only copies structure, not underlying data - changes to data affect both). Always use copy() when modifying a subset to avoid SettingWithCopyWarning. Shallow copies are rarely needed.",
            "Hard",
            90
        ),
        create_question(
            "What does df.value_counts() do when applied to a Series?",
            [
                "Counts total values",
                "Returns counts of unique values in descending order",
                "Calculates the sum",
                "Finds the maximum value"
            ],
            1,
            "value_counts() returns a Series with counts of unique values, sorted by count (descending). Usage: df['category'].value_counts(). Useful for categorical data analysis. normalize=True returns proportions instead of counts. dropna=False includes NaN in counts. For DataFrame, use df.value_counts() (counts unique rows) or apply to specific columns.",
            "Medium",
            75
        ),
        create_question(
            "What is the purpose of df.astype() in Pandas?",
            [
                "To delete columns",
                "To convert DataFrame column types to specified dtype",
                "To sort the DataFrame",
                "To filter rows"
            ],
            1,
            "astype() casts columns to specified data types. Usage: df['col'] = df['col'].astype('int64') or df = df.astype({'col1': 'int32', 'col2': 'float64'}). Common conversions: to numeric ('int', 'float'), to category ('category' for memory efficiency), to string ('str'). errors='ignore' prevents raising errors. Proper dtypes improve memory usage and performance.",
            "Medium",
            80
        ),
        create_question(
            "What does df.query('price > 100 and category == \"electronics\"') do?",
            [
                "Deletes matching rows",
                "Filters rows using a SQL-like string expression",
                "Updates the DataFrame",
                "Creates a new column"
            ],
            1,
            "query() filters rows using a string expression that can reference column names directly. Cleaner syntax than boolean indexing for complex conditions. Equivalent to: df[(df['price'] > 100) & (df['category'] == 'electronics')]. Can use @variable for external variables. Supports and, or, not. More readable for complex filters, though boolean indexing is more flexible.",
            "Medium",
            85
        ),
        create_question(
            "In Pandas, what is the purpose of df.reset_index()?",
            [
                "To delete the index",
                "To reset index to default integer sequence, optionally moving current index to a column",
                "To sort by index",
                "To rename the index"
            ],
            1,
            "reset_index() replaces the current index with default RangeIndex (0, 1, 2, ...). By default, old index becomes a column. drop=True discards old index. Useful after filtering/grouping when index becomes non-sequential or you want to discard a hierarchical index. inplace=True modifies in place. Opposite: set_index('col') makes a column the index.",
            "Medium",
            80
        ),
        create_question(
            "What does pd.get_dummies(df['category']) do?",
            [
                "Creates dummy rows",
                "Performs one-hot encoding: converts categorical variable into binary columns",
                "Removes the category column",
                "Generates random data"
            ],
            1,
            "get_dummies() creates one-hot encoding: for k categories, creates k binary (0/1) columns. For category=['A', 'B', 'A'], creates columns 'A' and 'B' with [1,0,1] and [0,1,0]. Essential for using categorical data in ML models. drop_first=True removes one category to avoid multicollinearity. Can apply to entire DataFrame: pd.get_dummies(df) encodes all object columns.",
            "Medium",
            85
        ),
        create_question(
            "What is the difference between df.describe() and df.info()?",
            [
                "They are identical",
                "describe() shows statistical summary of numeric columns; info() shows DataFrame structure and dtypes",
                "describe() is for strings only",
                "info() shows statistics"
            ],
            1,
            "describe() provides statistical summary (count, mean, std, min, quartiles, max) for numeric columns. include='all' includes non-numeric. info() shows: number of rows, column names, non-null counts, dtypes, memory usage. Use describe() for data distribution, info() for structure and missing data overview. Both are essential for initial data exploration.",
            "Medium",
            75
        )
    ]
    return questions


def populate_numpy():
    """15 NumPy Questions"""
    questions = [
        create_question(
            "What is the difference between np.array([1, 2, 3]) and np.array([[1, 2, 3]])?",
            [
                "They are identical",
                "First is 1D (shape (3,)); second is 2D (shape (1, 3))",
                "First is faster",
                "Second stores more data"
            ],
            1,
            "Shape matters! [1,2,3] creates 1D array with shape (3,). [[1,2,3]] creates 2D array with shape (1, 3) - one row, three columns. This affects operations: 1D arrays don't have explicit row/column orientation, while 2D do. For matrix operations, 2D is often needed. Check with arr.shape. Use arr.reshape() to convert between shapes.",
            "Medium",
            85
        ),
        create_question(
            "What does NumPy broadcasting allow you to do?",
            [
                "Transmit data over network",
                "Perform operations on arrays of different shapes by automatically expanding them",
                "Increase array size",
                "Parallelize computations"
            ],
            1,
            "Broadcasting allows arithmetic operations on arrays of different shapes without explicit replication. Rules: (1) Dimensions are aligned from right, (2) Dimensions of size 1 are stretched, (3) Dimensions must match or be 1. Example: (3,1) + (4,) → both broadcast to (3,4). Enables efficient memory usage and cleaner code. Understanding broadcasting is key to vectorized NumPy operations.",
            "Hard",
            95
        ),
        create_question(
            "What is the difference between np.dot(A, B) and A * B for 2D arrays?",
            [
                "They are identical",
                "np.dot() performs matrix multiplication; * performs element-wise multiplication",
                "np.dot() is deprecated",
                "* performs matrix multiplication"
            ],
            1,
            "* (or np.multiply()) is element-wise multiplication: corresponding elements are multiplied. Requires same shape. np.dot() (or @ operator in Python 3.5+) performs matrix/dot product: (m,n) @ (n,p) → (m,p). For 1D arrays, dot is inner product. For higher dimensions, see np.matmul(). Use @ for matrix multiplication, * for element-wise.",
            "Medium",
            90
        ),
        create_question(
            "What does arr.reshape(-1, 1) do?",
            [
                "Deletes the array",
                "Reshapes to column vector: infers first dimension, sets second to 1",
                "Transposes the array",
                "Flattens the array"
            ],
            1,
            "reshape(-1, 1) creates a column vector. -1 means 'infer this dimension'. For arr with 6 elements, (-1,1) becomes (6,1). (-1,) or .flatten() creates 1D. (1,-1) creates row vector. reshape doesn't copy data (returns view) if possible. reshape(-1) is common for flattening multi-dimensional arrays to 1D. Must preserve total element count.",
            "Medium",
            85
        ),
        create_question(
            "What is the purpose of np.where(condition, x, y)?",
            [
                "To find array indices",
                "Returns array with elements from x where condition is True, from y where False",
                "To filter array",
                "To sort array"
            ],
            1,
            "np.where() is vectorized if-else: where condition is True, take from x; else take from y. Example: np.where(arr > 0, arr, 0) replaces negative values with 0. With just condition, np.where(condition) returns indices where True (tuple of arrays). Powerful for conditional operations without loops. Similar to array[condition] = value for boolean indexing.",
            "Hard",
            90
        ),
        create_question(
            "What does np.random.seed(42) do?",
            [
                "Plants random numbers",
                "Sets the random number generator seed for reproducibility",
                "Generates 42 random numbers",
                "Deletes random state"
            ],
            1,
            "Setting seed ensures reproducible random numbers. Same seed → same sequence. Essential for debugging and reproducibility in ML experiments. Use before random operations: np.random.seed(42); np.random.randn(5) always gives same 5 numbers. Modern approach: rng = np.random.default_rng(42); rng.random() for better random generation and isolation.",
            "Medium",
            75
        ),
        create_question(
            "What is the difference between np.sum(arr, axis=0) and np.sum(arr, axis=1) for a 2D array?",
            [
                "They are identical",
                "axis=0 sums down columns (row-wise sum); axis=1 sums across rows (column-wise sum)",
                "axis=0 is faster",
                "axis=1 is deprecated"
            ],
            1,
            "axis specifies which dimension to collapse. For 2D array (rows, cols): axis=0 aggregates along rows (down columns), returning one value per column. axis=1 aggregates along columns (across rows), returning one value per row. Think of axis as the dimension that disappears. Same logic for mean, max, etc. No axis means aggregate all elements.",
            "Hard",
            95
        ),
        create_question(
            "What does np.arange(0, 10, 2) create?",
            [
                "Array [0, 2, 4, 6, 8]",
                "Array [0, 1, 2, ... 10]",
                "Array [2, 4, 6, 8, 10]",
                "Array [0, 10, 2]"
            ],
            0,
            "np.arange(start, stop, step) creates array from start to stop (exclusive) with step. Here: [0, 2, 4, 6, 8]. Similar to Python's range() but returns NumPy array. For floats, prefer np.linspace(start, stop, num) which includes stop and specifies count instead of step, avoiding floating-point issues. arange is half-open [start, stop).",
            "Medium",
            75
        ),
        create_question(
            "What is the purpose of np.newaxis in NumPy?",
            [
                "Creates a new array",
                "Adds a new dimension of size 1 to an array",
                "Deletes an axis",
                "Transposes the array"
            ],
            1,
            "np.newaxis (or None) increases dimensionality by adding axis of size 1. arr[np.newaxis, :] converts (n,) to (1,n). arr[:, np.newaxis] converts to (n,1). Useful for broadcasting: (3,) and (4,) can't broadcast, but (3,1) and (4,) → (3,4). Cleaner than reshape for adding dimensions. Essential for proper broadcasting in matrix operations.",
            "Hard",
            90
        ),
        create_question(
            "What does arr.flatten() vs arr.ravel() do?",
            [
                "They are always identical",
                "flatten() always copies; ravel() returns view if possible (faster but changes affect original)",
                "ravel() is deprecated",
                "flatten() is faster"
            ],
            1,
            "Both convert multi-dimensional array to 1D. ravel() returns a view when possible (contiguous memory) - modifications affect original. flatten() always returns a copy - safe to modify. ravel() is faster and memory-efficient for large arrays. reshape(-1) is like ravel(). Use flatten() when you need an independent copy, ravel() for efficiency when view is acceptable.",
            "Hard",
            95
        ),
        create_question(
            "What is the purpose of np.concatenate([arr1, arr2], axis=0)?",
            [
                "Multiplies arrays",
                "Joins arrays along specified axis (axis=0 means vertically/row-wise)",
                "Finds common elements",
                "Splits arrays"
            ],
            1,
            "concatenate() joins arrays along existing axis. axis=0 stacks vertically (appends rows), axis=1 horizontally (appends columns). Arrays must have compatible shapes (all dimensions except concatenation axis must match). Alternatives: np.vstack() (vertical), np.hstack() (horizontal), np.stack() (creates new axis). concatenate is more general and flexible.",
            "Medium",
            85
        ),
        create_question(
            "What does np.argmax(arr) return?",
            [
                "The maximum value",
                "The index of the maximum value",
                "An array of all maximum values",
                "The count of maximum values"
            ],
            1,
            "argmax() returns the index (not value) of the maximum element. For 1D: single integer. For multi-D without axis: flattened index. With axis: indices along that axis. Example: arr = [3,1,4,2]; np.argmax(arr) = 2. For value, use arr.max() or arr[np.argmax(arr)]. Similarly, argmin() for minimum index. Useful for classification (getting predicted class).",
            "Medium",
            80
        ),
        create_question(
            "What is the purpose of np.linspace(0, 1, 5)?",
            [
                "Creates array [0, 1, 2, 3, 4]",
                "Creates array [0.0, 0.25, 0.5, 0.75, 1.0] - 5 evenly spaced values from 0 to 1",
                "Creates 5 random numbers between 0 and 1",
                "Creates array [0, 1, 0, 1, 0]"
            ],
            1,
            "linspace(start, stop, num) creates num evenly spaced values from start to stop (inclusive). Here: [0, 0.25, 0.5, 0.75, 1.0]. Unlike arange (uses step), linspace uses count. endpoint=False excludes stop. Useful for plotting, creating grids. Preferred over arange for floats to avoid precision issues. Stop is included by default.",
            "Medium",
            80
        ),
        create_question(
            "What does arr[:, ::-1] do for a 2D array?",
            [
                "Transposes the array",
                "Reverses columns (flips horizontally)",
                "Reverses rows",
                "Deletes last column"
            ],
            1,
            "Slicing with negative step reverses. [:, ::-1] means: all rows (:), all columns reversed (::-1). This flips columns horizontally. [::-1, :] reverses rows (vertical flip). [::-1, ::-1] reverses both. These create views (no copy). Important: negative step creates reversed view efficiently without copying data.",
            "Medium",
            85
        ),
        create_question(
            "What is the difference between np.copy(arr) and arr.view()?",
            [
                "They are identical",
                "copy() creates independent copy; view() creates new array object sharing same data",
                "view() is faster and always preferred",
                "copy() shares data"
            ],
            1,
            "copy() creates independent deep copy - changes don't affect original. view() creates new array object but shares underlying data - changes affect both. Slicing usually creates views. Use copy() when you need independence. view() is memory-efficient but requires care. To check: arr.base is None for copy, refers to original for view. Assignment (b=a) creates reference (not even a view).",
            "Hard",
            95
        )
    ]
    return questions


if __name__ == "__main__":
    db = QuestionDatabase()

    print("Populating Pandas questions...")
    db.add_bulk_questions("Pandas", populate_pandas())
    print(f"✓ Added {len(populate_pandas())} Pandas questions")

    print("Populating NumPy questions...")
    db.add_bulk_questions("NumPy", populate_numpy())
    print(f"✓ Added {len(populate_numpy())} NumPy questions")

    stats = db.get_statistics()
    print(f"\n{'='*60}")
    print(f"BATCH 3 COMPLETE - Data Libraries")
    print(f"{'='*60}")
    print(f"Total questions in database: {stats['total_questions']}")
    print("\nBatch 3 questions by category:")
    for category in ["Pandas", "NumPy"]:
        count = db.get_question_count(category)
        print(f"  {category}: {count} questions")
    print(f"\nDatabase saved to: questions_db.json")
