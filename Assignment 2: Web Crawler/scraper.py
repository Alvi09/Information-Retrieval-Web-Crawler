import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from collections import defaultdict
from utils import get_logger

visited_urls = set()
longest_page = {"url": "", "length": 0}
top_50 = defaultdict(int)
subdomains = defaultdict(int)

logger = get_logger("CRAWLER")

def scraper(url, resp):
    links = extract_next_links(url, resp)
    get_analytics()
    return list(links)


def extract_next_links(url, resp):
    # Implementation required.
    # url: the URL that was used to get the page
    # resp.url: the actual url of the page
    # resp.status: the status code returned by the server. 200 is OK, you got the page. Other numbers mean that there was some kind of problem.
    # resp.error: when status is not 200, you can check the error here, if needed.
    # resp.raw_response: this is where the page actually is. More specifically, the raw_response has two parts:
    #         resp.raw_response.url: the url, again
    #         resp.raw_response.content: the content of the page!
    # Return a list with the hyperlinks (as strings) scrapped from resp.raw_response.content
    links = set()
    if resp and resp.status == 200 and resp.raw_response and resp.raw_response.content:
        content = BeautifulSoup(resp.raw_response.content, "html.parser")
        if check(url, resp):
            for a_tag in content.findAll("a"):
                href = a_tag.attrs.get("href")
                if href:
                    temp = is_valid(href)
                    if temp:
                        links.add(temp)
                        visited_urls.add(temp)
    return links


def is_valid(url):
    # Decide whether to crawl this url or not. 
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.
    try:
        if url.find("#") == -1:
            new_url = url
        else:
            new_url = url[:url.find("#")]

        if new_url in visited_urls:
            return False
        
        if len(new_url) > 150:
            return False
        
        parsed = urlparse(url)

        if parsed.scheme not in set(["http", "https"]):
            return False
        
        query_bl = {"replytocom", "share", "page_id", "afg", "ical", "action"}
        if any([(query in parsed.query) for query in query_bl]):
            return False

        domain_wl = ["ics.uci.edu", "cs.uci.edu", "informatics.uci.edu", "stat.uci.edu", "today.uci.edu/department/information_computer_sciences"]
        if not any([(domain in parsed.netloc) for domain in domain_wl]):
            return False

        if "eecs.uci.edu" in parsed.netloc:
            return False
        
        path_bl = {"/events/", "/day/", "/week/", "/month/", "/list/", "?filter"}
        if any([(path in url.lower()) for path in path_bl]):
            return False

        if re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            + r"|png|tiff?|mid|mp2|mp3|mp4"
            + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            + r"|epub|dll|cnf|tgz|sha1"
            + r"|thmx|mso|arff|rtf|jar|csv"
            + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$", parsed.path.lower()):
            return False

        subdomains[parsed.netloc] += 1

        return new_url
        
    except TypeError:
        print ("TypeError for ", parsed)
        raise

def check(url, resp):
    if url.find("#") == -1:
        new_url = url
    else:
        new_url = url[:url.find("#")]

    content = BeautifulSoup(resp.raw_response.content, "html.parser")
    textlist = content.text.split("\n")
    text = []
    stopwords = {"ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", 
        "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", 
        "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", 
        "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", 
        "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", 
        "yourselves",  "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", 
        "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", 
        "a", "by", "doing", "it", "how", "further", "was", "here", "than", "d", "b"}
    for i in textlist:
        temp = re.findall(r'[0-9a-z]+', i.lower())
        text.extend(temp)

    if len(text) < 50:
        return False

    count = 0
    for i in text:
        if len(i) >= 3 and i not in stopwords:
            top_50[i] += 1
            count += 1
    
    if longest_page["length"] < count:
        longest_page["url"] = new_url
        longest_page["length"] = count
    
    return True


def get_analytics():

    logger.info(f"Longest page: {longest_page['url']}, {longest_page['length']}")
    logger.info(f"Top 50 Words: {sorted(top_50.items(), key = lambda x: -x[1])[:50]}")
    logger.info(f"Subdomains: {sorted(subdomains.items())}, number of subdomains: {len(subdomains)}")
    logger.info(f"Number of unique pages: {len(visited_urls)}")

if __name__ == "__main__":
    print(scraper("https://www.ics.uci.edu", None))
