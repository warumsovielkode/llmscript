Author: Kevin Zhang 张明凯 2024470359

Date: 7/8/2024

Process:

1. Import pandas and create a dataframe that will store the input csv file
2. Define two functions that will operate on each row to check if there are too many hashes or bullets. The rows will not be returned to the output if there are more hashes or bullets than the threshold
3. Define a function that checks if a row has two or more stop words. If a row does, it will not be returned to the output.
4. Output the cleaned data to the output file path
    