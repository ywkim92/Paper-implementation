import os
import re
import copy
import subprocess as sp

def change_url():
    gitfiles = sp.getoutput('git ls-files')
    gitfiles = [g.replace(' ','%20') for g in gitfiles.split('\n')]
    gitfiles_nb = [g for g in gitfiles if re.search(r'(?<=/)[^/\.]+\.ipynb',g) is not None]
    change_url_dict = dict(zip( [re.search(r'(?<=/)[^/\.]+\.ipynb',g).group(0) for g in gitfiles_nb] , gitfiles_nb))
    return change_url_dict
    
def update_readme( ):
    readme_path = 'C:\\Users\\ywkim\\Github\\Paper implementation\\README.md'
    github_url = 'https://github.com/ywkim92/Paper-implementation/blob/main/'
    notebook_viewer = 'https://nbviewer.org/github/ywkim92/Paper-implementation/blob/main/'
    
    change_url_dict_ = change_url()
    change_url_dict_copy = copy.deepcopy(change_url_dict_)
    
    lines = []
    with open(readme_path) as file:
        for f in file:
            if re.search(r'(?<=main/)[^)]+\.ipynb(?=\))', f) is not None:
                f_cur_url = re.search(r'(?<=main/)[^)]+\.ipynb(?=\))', f).group(0)
                f_keyword = re.search(r'[^)/]+\.ipynb$', f_cur_url).group(0)
                change_url_dict_copy.pop(f_keyword)
                
                if f_cur_url != change_url_dict_[f_keyword]:
                    print('* url changed', f_keyword)
                    f = f.replace(f_cur_url, change_url_dict_[f_keyword])
            f = f.replace(github_url, notebook_viewer)
            
            lines.append(f)
    
    
#   file_list = [g for g in sp.getoutput('git ls-files --others --exclude-standard').split('\n') if g.endswith('.ipynb')]
    file_list = list(change_url_dict_copy.keys())
    print(file_list)
    add_list = []
    for name in file_list:
        re_search = re.search(r'[\w\-/ %]+\.ipynb$', name)
        if re_search is None: continue
        else:
            re_search_word = re.search(r'[^/]+(?=\.ipynb)', re_search.group(0)).group(0)
            print(re_search_word)
            re_search_url = notebook_viewer + change_url_dict_copy[name].replace(' ','%20')
            
            if len(re_search_word)<7: re_search_word_input = re_search_word.upper()
            else: re_search_word_input = re_search_word.capitalize()
            add_str = '1. [{}]({})\n'.format( re_search_word_input  , re_search_url )
            
            add_list.append(add_str)
    print('newly added untracked files:',add_list)
    
    
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
    os.chdir('C:\\Users\\ywkim\\Github\\Paper implementation')
    update_readme( )