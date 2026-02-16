def tokenize(text):
    """
    Tokenizes the input text by splitting on whitespace and converting to lowercase.
    
    :param text: The input text to tokenize.
    :return: A list of tokens.
    """
    return text.lower().split()

def preprocess_data(df):
    """
    Preprocesses the input DataFrame by tokenizing the 'description' column and creating a new 'tokens' column.
    
    :param df: The input DataFrame with a 'description' column.
    :return: A new DataFrame with an additional 'tokens' column.
    """
    df['tokens'] = df['description'].apply(tokenize)
    return df
