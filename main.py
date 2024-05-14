
import content_features as ctnfe
import url_features as urlfe
import external_features as trdfe
import pandas as pd 
import urllib.parse
import tldextract
import requests
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import streamlit as st
from requests.exceptions import HTTPError, RequestException, Timeout
key = "cgsggo8gwogwc4osc40cwoko04gs0840skwco80c"
def is_URL_accessible(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}  # Example of headers, you may need to customize this
        page = requests.get(url, headers=headers, timeout=5)
        page.raise_for_status()  # Raise an exception for HTTP errors (status codes other than 2xx)
        return True, page
    except requests.exceptions.Timeout:
        print("Request timed out. Unable to access URL:", url)
    except HTTPError as e:
        print("HTTP Error:", e)
        print("Unable to access URL:", url)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)
    return False, None




import tldextract
import urllib.parse

# Cache pour tldextract
tldextract_cache = {}

def get_domain(url):
    o = urllib.parse.urlsplit(url)
    if o.hostname is None:
        return None, None, None, None, None

    subdomain = o.hostname.split('.')[0]

    # Utilisation du cache pour tldextract
    if url in tldextract_cache:
        extracted_domain = tldextract_cache[url]
    else:
        extracted_domain = tldextract.extract(url)
        tldextract_cache[url] = extracted_domain

    domain = extracted_domain.domain + '.' + extracted_domain.suffix
    tld = extracted_domain.suffix

    return o.hostname, domain, subdomain, o.path, tld


import multiprocessing
import functools

class TimedOutExc(Exception):
    pass

def target_func(queue, func, args, kwargs):
    try:
        result = func(*args, **kwargs)
        queue.put(('result', result))
    except Exception as e:
        queue.put(('exception', e))

def deadline(timeout):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=target_func, args=(queue, func, args, kwargs))
            process.start()
            process.join(timeout)
            if process.is_alive():
                process.terminate()
                process.join()
                raise TimedOutExc("Function call timed out")
            else:
                result_type, result_value = queue.get()
                if result_type == 'result':
                    return result_value
                elif result_type == 'exception':
                    raise result_value
        return wrapper
    return decorate




def getPageContent(url):
    parsed = urlparse(url)
    url = parsed.scheme+'://'+parsed.netloc
    try:
        page = requests.get(url)
    except:
        if not parsed.netloc.startswith('www'):
            url = parsed.scheme+'://www.'+parsed.netloc
            page = requests.get(url)
    if page.status_code != 200:
        return None, None
    else:    
        return url, page.content
def extract_data_from_URL(url, content, domain, hostname):
   
    href = {'internals': [], 'externals': [], 'null': []}
    link = {'internals': [], 'externals': [], 'null': []}
    anchor = {'safe': [], 'unsafe': []}
    media = {'internals': [], 'externals': []}
    form = {'internals': [], 'externals': [], 'null': []}
    css = {'internals': [], 'externals': [], 'null': []}
    favicon = {'internals': [], 'externals': [], 'null': []}
    iframe = {'visible': [], 'invisible': []}
    title = None
    text = None

    soup = BeautifulSoup(content, 'html.parser', from_encoding='iso-8859-1')

    
    for tag in soup.find_all(['a', 'link', 'script'], href=True):
        if 'href' in tag.attrs:
            href_type = 'internals' if tag['href'].startswith('/') else 'externals'
            href[href_type].append(tag['href'])


    for tag in soup.find_all(['img', 'audio', 'embed', 'iframe'], src=True):
        media_type = 'internals' if tag['src'].startswith('/') else 'externals'
        media[media_type].append(tag['src'])

  
    for form_tag in soup.find_all('form', action=True):
        form_type = 'internals' if form_tag['action'].startswith('/') else 'externals'
        form[form_type].append(form_tag['action'])


    for css_tag in soup.find_all(['link', 'style'], {'rel': 'stylesheet'}):
        css_type = 'internals' if css_tag['href'].startswith('/') else 'externals'
        css[css_type].append(css_tag['href'])


    for link_tag in soup.find_all('link', rel='icon'):
        favicon_type = 'internals' if link_tag['href'].startswith('/') else 'externals'
        favicon[favicon_type].append(link_tag['href'])
    for iframe_tag in soup.find_all('iframe'):
        if iframe_tag.get('width') == "0" and iframe_tag.get('height') == "0":
            iframe['invisible'].append(iframe_tag['src'])
        else:
            iframe['visible'].append(iframe_tag['src'])

    title = soup.title.string if soup.title else None


    text = soup.get_text()

    return href, link, anchor, media, form, css, favicon, iframe, title, text

def words_raw_extraction(domain, subdomain, path):
 
    domain_parts = domain.split('.') if domain else []
    subdomain_parts = subdomain.split('.') if subdomain else []
    path_parts = path.split('/') if path else []
  
    words_raw = domain_parts + subdomain_parts + path_parts
    words_raw_host = domain_parts + subdomain_parts
    words_raw_path = path_parts
    
    return words_raw, words_raw_host, words_raw_path
import pickle
import numpy as np
import streamlit as st
import time


loaded_model = pickle.load(open("xgb_model.pkl", 'rb'))

def FeatureExtraction(url):
    state, page = is_URL_accessible(url)
    if state:
        content = page.content
        hostname, domain, subdomain, path, tld = get_domain(url)
        
        if hostname is None or domain is None:
            print("Unable to extract hostname and domain from the URL:", url)
            return None

        Href, Link, Anchor, Media, Form, CSS, Favicon, IFrame, Title, Text = extract_data_from_URL(
            url, content, domain, hostname)
        
        words_raw, words_raw_host, words_raw_path = words_raw_extraction(domain, subdomain, path)

        function_names = [
            urlfe.url_length(url),
            urlfe.url_length(hostname),
            urlfe.having_ip_address(url),
            urlfe.count_dots(hostname),
            urlfe.count_hyphens(url),
            urlfe.count_at(url),
            urlfe.count_exclamation(url),
            urlfe.count_and(url),
            urlfe.count_equal(url),
            urlfe.count_underscore(url),
            urlfe.count_tilde(url),
            urlfe.count_percentage(url),
            urlfe.count_slash(url),
            urlfe.count_colon(url),
            urlfe.count_comma(url),
            urlfe.count_semicolumn(url),
            urlfe.count_space(url),
            urlfe.check_www(words_raw),
            urlfe.check_com(words_raw),
            urlfe.count_double_slash(url),
            urlfe.count_http_token(path),
            urlfe.https_token(url),
            urlfe.ratio_digits(url),
            urlfe.ratio_digits(hostname),
            urlfe.port(url),
            urlfe.tld_in_path(tld, path),
            urlfe.tld_in_subdomain(domain, hostname),
            urlfe.abnormal_subdomain(url),
            urlfe.count_subdomain(hostname),
            urlfe.prefix_suffix(url),
            urlfe.shortening_service(url),
            urlfe.count_redirection(page),
            urlfe.length_word_raw(hostname, path),
            urlfe.char_repeat(words_raw),
            urlfe.shortest_word_length(words_raw),
            urlfe.shortest_word_length(words_raw_host),
            urlfe.shortest_word_length(words_raw_path),
            urlfe.longest_word_length(words_raw),
            urlfe.longest_word_length(words_raw_host),
            urlfe.longest_word_length(words_raw_path),
            urlfe.average_word_length(words_raw),
            urlfe.average_word_length(words_raw_host),
            urlfe.average_word_length(words_raw_path),
            urlfe.phish_hints(path),
            urlfe.domain_in_brand(domain),
            urlfe.domain_in_brand1(domain),
            urlfe.brand_in_path(domain, path),
            urlfe.suspecious_tld(domain),
            urlfe.statistical_report(url, domain),
            ctnfe.nb_hyperlinks(Href, Link, Media, Form, CSS, Favicon),
            ctnfe.internal_hyperlinks(Href, Link, Media, Form, CSS, Favicon),
            ctnfe.external_hyperlinks(Href, Link, Media, Form, CSS, Favicon),
            ctnfe.external_css(CSS),
            ctnfe.external_redirection(Href, Link, Media, Form, CSS, Favicon),
            ctnfe.external_errors(Href, Link, Media, Form, CSS, Favicon),
            ctnfe.login_form(Form),
            ctnfe.external_favicon(Favicon),
            ctnfe.links_in_tags(Link),
            ctnfe.internal_media(Media),
            ctnfe.external_media(Media),
            ctnfe.iframe(IFrame),
            ctnfe.popup_window(Text),
            ctnfe.safe_anchor(Anchor),
            ctnfe.onmouseover(Text),
            ctnfe.right_clic(Text),
            ctnfe.empty_title(Title),
            ctnfe.domain_in_title(domain,Title),
            ctnfe.domain_with_copyright(domain,Text),
            trdfe.whois_registered_domain(url),
            trdfe.domain_registration_length(domain),
            trdfe.domain_age(domain),
            trdfe.web_traffic(url),
            trdfe.dns_record(domain),
            trdfe.google_index(url),
            trdfe.page_rank(key,domain)
        ]
        
        return function_names
    else:
        print("URL is not accessible:", url)
        return None

def prediction(features):
    try:
   
        prediction = loaded_model.predict(features)
        if prediction[0]== 1:
            return "Attention! This web page is a potential PHISHING!"
        else:
            return "This web page seems legitimate!"
    except ValueError as e:
        print("ValueError:", e)
        return "Error occurred during prediction. Please check input data."
import streamlit as st
import time
import numpy as np

def main():
 
    st.title('Welcome to the Phishing Detection Tool')
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #add8e6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title('EXAMPLE PHISHING URLs:')
    st.sidebar.write('http://www.u1903007.cp.regruhosting.ru/')
    st.sidebar.write('https://meta-case-id8759235.meta-appeals.com/1.php')
    st.sidebar.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE! SO, THE EXAMPLES SHOULD BE UPDATED!')

    st.write("## Enter the URL to check for phishing")

    # Input URL
    url = st.text_input('Enter the URL')

    # Button to trigger prediction
    if st.button('Check!'):
        # Perform feature extraction
        features = FeatureExtraction(url)
        print("Features:", features)  # Print features for debugging
        
        if features is not None:
            # Reshape features
            features_as_numpy_array = np.asarray(features)
            features_reshaped = features_as_numpy_array.reshape(1, -1)
            
            # Get prediction
            result = prediction(features_reshaped)

            # Display result
            if result == "This web page seems legitimate!":
                st.success(result)
                st.balloons()
                time.sleep(1)
                st.balloons()
                time.sleep(1)
                st.balloons()
            elif result == "Attention! This web page is a potential PHISHING!":
                st.warning(result)
                st.snow()
                time.sleep(1)
                st.snow()
                time.sleep(1)
                st.snow()
        else:
            st.error("Feature extraction failed. Please check the URL.")

    # Signature at the bottom
    st.markdown(
        """
        <style>
        .signature {
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
        }
        </style>
        """
        "<div class='signature'>enis.tn</div>",
        unsafe_allow_html=True,
    )

if __name__ == '__main__':
    main()
