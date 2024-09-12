import os
import chardet
import csv

column = ['Sample','#','Year','Month','Day','Hour','#','#','#','Lat','Lon','Lev','#','#','#','#','#','#','#','#','#','#']

def SwitchTxtToCsv(folder_path:str):
    file_list = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        file_list.append(item_path)
        
    for file_path in file_list:
        csv_file_path = "/mnt/d/software/hysplit/2023/csv03/"+ os.path.basename(file_path) +".csv"
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read()) 
        encoding = result['encoding']
        if os.path.exists(file_path):
            with open(file_path, 'r',encoding=encoding) as txt_file:
                for i in range(10):
                    next(txt_file)
                for line in txt_file:
                    lines = txt_file.readlines()           
            csv_content = ''

            csv_content += ','.join(column) + '\n'
            
            for line in lines:  
                columns = line.strip().split()
                csv_content += ','.join(columns) + '\n'
            with open(csv_file_path, 'a', encoding='utf-8') as csv_file:
                csv_file.write(csv_content)   
        else:
            print(f"文件 {file_path} 不存在")
    return

file_list = '/mnt/d/software/hysplit/2023/03'
SwitchTxtToCsv(file_list)
