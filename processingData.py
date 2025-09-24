from datasets import load_from_disk, load_dataset
import json
import os
from collections import deque
from bs4 import BeautifulSoup
import cssutils
from premailer import Premailer

VOID_TAGS = [
    'area','base','br','col','embed','hr','img','input',
    'link','meta','source','track','wbr','audio','video'
]

LAYOUT_NEUTRAL_TAGS = [
    'span','em','strong','abbr','cite','code','script',
    'mark','time','b','i','u','small','sup','sub'
]

class GeneratingGT:
    def __init__(self,ds):
        self.ds = ds
        self.len_of_ds = len(ds)
        self.output = []
        self.exceptions = []

    def inspectingHTML(self,text):
        # print(type(text))
        number_of_nodes = 0
        for ii in range(1,len(text)):
            if text[ii-1] == '<' and text[ii] != '/':
                number_of_nodes += 1
        return number_of_nodes

    def inspectingLayout(self,root_node):
        # print(root_node['type'])
        number_of_nodes = 1
        qq = deque()
        qq.append(root_node)
        while(len(qq)):
            node = qq.popleft()
            ss = ""
            if node is not None:
                number_of_nodes += len(node['children'])
                for child in node['children']:
                    ss += child['type']+' '
                    qq.append(child)
                    if child['style'] is not None and 'flex-direction: column' in child['style']:
                        # print()
                        # print(child['style'])
                        # print()
                        pass
            # print(ss)
        return number_of_nodes

    def parse_style(self,style_str):
        """Convert style string into a dictionary."""
        styles = {}
        if style_str:
            for item in style_str.split(";"):
                if ":" in item:
                    key, value = item.split(":", 1)
                    styles[key.strip()] = value.strip()
        return styles

    def generating0123(self,tag):

        if tag.name in VOID_TAGS or tag.name in LAYOUT_NEUTRAL_TAGS:
            return
        
        # print(f"<{tag.name}>")
        style_dict = self.parse_style(tag.get('style'))

        if style_dict:
            if 'display' in style_dict:
                if style_dict['display'] == 'flex' or \
                style_dict['display'] == 'inline-flex':
                    if 'flex-direction' in style_dict:
                        if style_dict['flex-direction'] == 'column':
                            self.output.append(1)
                        else:
                            self.output.append(0)
                    else:
                        self.output.append(0)
                elif style_dict['display'] == 'grid':
                    if 'grid-auto-flow' in style_dict:
                        if style_dict['grid-auto-flow'] == 'column':
                            self.output.append(0)
                        else:
                            self.output.append(1)
                    else:
                        self.output.append(1)
                else:
                    self.output.append(1)
            else:
                self.output.append(1)
        else:
            self.output.append(1)
                

        for child in tag.children:
            if child.name:
                self.generating0123(child)
        
        self.output.append(2)
        # print(f"</{tag.name}>")
    
    def testing(self):
        self.output = []
        with open("test.html", "r", encoding="utf-8") as f:
            html = f.read()
        inline_style = transform(html)
        pretty_html = BeautifulSoup(inline_style, 'html.parser')
        self.generating0123(pretty_html.html)

        with open("output.txt", "w", encoding="utf-8") as f:
            for ixx,item in enumerate(self.output):
                if ixx == 0:
                    f.write(str(item))
                else:
                    f.write('\n'+str(item))

    def generatingGT(self, ii, max_len=1401, end_marker=3):
        self.output = [4] ## 4 is BOS

        # Parse HTML
        soup = BeautifulSoup(ds[ii]['text'], "html.parser")

        # Remove all <link> tags
        for tag in VOID_TAGS+LAYOUT_NEUTRAL_TAGS:
            for link in soup.find_all(tag):
                link.decompose()

        # Convert back to string
        clean_html = str(soup)

        p = Premailer(clean_html, external_styles=False)
        try:
            inline_style = p.transform()
            pretty_html = BeautifulSoup(inline_style, 'html.parser')
            self.generating0123(pretty_html.html)
        except Exception as e:
            print(f"Exception is: {e}")
            print(f"\nException at: {ii}")
            self.exceptions.append(ii)
            # exit()
        

        self.output = self.output + [end_marker] * (max_len - len(self.output))

        return self.output

    def generatingDataset(self):

        os.makedirs('webcode2m_plc/image',exist_ok=True)
        os.makedirs('webcode2m_plc/code',exist_ok=True)
        os.makedirs('webcode2m_plc/layout',exist_ok=True)

        for ii in range(self.len_of_ds):
            layout = self.generatingGT(ii)

            self.save_html(ii,path='webcode2m_plc/code/'+f'{ii}.html')
            self.saveGT(ii,layout,path='webcode2m_plc/layout/'+f'{ii}.txt')
            self.save_an_image(ii,path='webcode2m_plc/image/'+f'{ii}.png')
        
        print("Dataset is created")

    def maxNodesIndex(self):
        max_nodes = -100
        max_index = -1
        for ii in range(self.len_of_ds):
            number_of_nodes = self.inspectingHTML(self.ds[ii]['text'])
            if max_nodes < number_of_nodes:
                max_nodes = number_of_nodes
                max_index = ii
        return max_nodes,max_index
    
    def minNodesIndex(self):
        min_nodes = 1e9
        min_index = -1
        for ii in range(self.len_of_ds):
            number_of_nodes = self.inspectingHTML(self.ds[ii]['text'])
            if min_nodes > number_of_nodes:
                min_nodes = number_of_nodes
                min_index = ii
        return min_nodes,min_index

    def save_an_image(self,ii,path=None):
        if path is None:
            path = f'{ii}.png'
        ds[ii]['image'].save(path)

    def sorted_tuples_of_number_of_nodes(self):
        tonon = []

        for ii in range(self.len_of_ds):
            tonon.append(ii,self.inspectingHTML(ds[ii]['text']))

        stonon = sorted(tonon, key = lambda x: x[1])

        return stonon
    
    def saveGT(self,ii,gt,path=None):
        if path is None:
            path = f'{ii}.txt'
        
        with open(path, "w", encoding="utf-8") as f:
            for ixx, item in enumerate(gt):
                if ixx == 0: f.write(str(item))
                else: f.write('\n'+str(item))

    def save_html(self,ii,path=None):
        if path is None:
            path = f'{ii}.html'

        soup = BeautifulSoup(ds[ii]['text'], "html.parser")
        pretty_html = soup.prettify()

        with open(path, "w", encoding="utf-8") as f:
            f.write(pretty_html)

    def howManyDivs(self,ii):
        return ds[ii]['text'].count('</div>')
    
    def howManySpans(self,ii):
        return ds[ii]['text'].count('</span>')

if __name__ == '__main__':
    ds = load_dataset("parquet", data_files="webcode_large_subset/*.parquet")["train"]
    
    ggt = GeneratingGT(ds)

    print(f"Number of Samples in the dataset: {ggt.len_of_ds}")

    # ggt.save_html(ii=456,path='actual_456.html')
    # print(ggt.howManySpans(ii=456))
    # ggt.generating0123(456)
    # ggt.testing()
    ggt.generatingDataset()
    print(f"Exceptions at the following numbers: {ggt.exceptions}")
    # ggt.save_html(538)