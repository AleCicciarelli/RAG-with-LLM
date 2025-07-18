import pandas as pd
import os

def infer_column_type(series):
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(series):
        return "DECIMAL"
    else:
        return "VARCHAR"

def is_unique_and_not_null(series):
    return series.is_unique and not series.isnull().any()

def infer_schema_from_csv(folder_path, output_file):
    tables = {}

    # Step 1: parse all CSVs
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            table_name = os.path.splitext(file)[0].upper()
            df = pd.read_csv(os.path.join(folder_path, file))
            schema = []
            primary_key = None

            for col in df.columns:
                col_upper = col.upper()
                col_type = infer_column_type(df[col])

                if is_unique_and_not_null(df[col]) and primary_key is None:
                    schema.append(f"    {col_upper} {col_type} PRIMARY KEY")
                    primary_key = col_upper
                else:
                    schema.append(f"    {col_upper} {col_type}")

            tables[table_name] = {"schema": schema, "df": df}

    # Step 2: try to infer foreign keys
    for table, info in tables.items():
        for other_table, other_info in tables.items():
            if table == other_table:
                continue
            other_columns = [col.upper() for col in other_info["df"].columns]
            for i, line in enumerate(info["schema"]):
                col_name = line.strip().split()[0]
                if col_name.endswith("KEY") and col_name in other_columns:
                    info["schema"][i] = f"    {col_name} {infer_column_type(info['df'][col_name.lower()])} REFERENCES {other_table}({col_name})"

    # Step 3: write to file
    with open(output_file, "w") as f:
        for table, info in tables.items():
            f.write(f"{table}(\n")
            f.write(",\n".join(info["schema"]))
            f.write("\n)\n\n")

# Esegui
infer_schema_from_csv("csv_data", "schemaTOY.txt")
