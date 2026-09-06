#actual loading of the file with diff extensions (csv anf excel). pandas for data reading


import pandas as pd 

def load_file(file_path : str, file_type : str):

    if file_type == ".csv":
        return pd.read_csv(file_path)
    
    elif file_type == ".xlsx":
        return pd.read_xlsx(file_path)
    
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Supported file types are: .csv and .xlsx")
    

## FOR NOW THIS IS ENOUGH, IN FUTURE WE CAN ADD MORE FILE TYPES LIKE JSON, XML ETC. AND ALSO ADD MORE OPTIONS FOR READING THE FILES LIKE ENCODING, DELIMITER ETC.
