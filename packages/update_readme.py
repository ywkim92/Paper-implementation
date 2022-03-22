import re
import os

def update_readme(file_list, ):
    readme_path = 'C:\\Users\\ywkim\\Github\\Paper implementation\\README.md'
    github_url = 'https://github.com/ywkim92/Paper-implementation/blob/main/'
    
    add_list = []
    for name in file_list:
        re_search = re.search(r'[\w\-]+\.ipynb$', name)
        if re_search is None: continue
        else: 
            add_str = '1. [{}]({})\n'.format( re.search(r'^.+(?=\.ipynb)', re_search.group(0)).group(0).capitalize()  , github_url + re_search.group(0)  )
            add_list.append(add_str)
    
    lines = []
    with open(readme_path) as file:
        for f in file:
            lines.append(f)
    
    division_idx = lines.index('- - -  \n')
    lines_pre = lines[:division_idx+1]
    lines_post = lines[division_idx+1:]
    lines_post_new = [l for l in lines_post if l!='\n']+add_list
    lines_post_sorted = sorted(lines_post_new, key=lambda x: re.search(r'(?<=^\d\. \[)[^\[\]]+(?=\])', x).group().lower())
    
    updated_list = lines_pre + ['\n'.join(lines_post_sorted)]
    updated_str = ''.join(updated_list)
    
    with open(readme_path,'w') as file:
        file.write(updated_str)
        
        
        
if __name__=='__main__':
    file_list = input().split(', ')
    update_readme(file_list, )