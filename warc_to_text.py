# converts warc file into a pandas dataframe
# each row of dataframe contains url,html,text for a specific html file

from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
import pandas as pd
import re

def get_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.body
    if body is None:
        return None
    for tag in body.select('script'):
        tag.decompose()
    for tag in body.select('style'):
        tag.decompose()
    text = body.get_text(separator='\n')
    return text

def process_warc(file_path):
    rows = []
    with open(file_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'response':
                url = record.rec_headers.get_header('WARC-Target-URI')
                html_raw = record.content_stream().read()#.decode('utf-8')
                html = html_raw.decode('utf-8')
                html = re.sub('\n+','\n',html)
                html = re.sub('[\r\f\v]','',html)
                html = re.sub('\t+',' ',html)
                text = get_text(html_raw)
                if isinstance(text,type(None)):
                    print(html_raw)
                    continue
                text = re.sub('\n+','. ',text)
                text = re.sub('[\r\f\v]','',text)
                text = re.sub('\t+',' ',text)
                text = re.sub('[ ]{2,}',' ', text)
                rows.append([url,html,text])
    df = pd.DataFrame(data=rows,columns=['url','html','text'])
    return df
    
if __name__ == '__main__':
    warc = '/Users/ryankingery/desktop/6_Shooting_Douglas_2018/Shooting_Douglas_2018.warc'
    repo_path = '/Users/ryankingery/Repos/Big-Data-Text-Summarization/'
    df = process_warc(warc)
    #df.to_csv(repo_path+'text.csv')