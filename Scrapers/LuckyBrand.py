import json
import os
import pickle
import time
import urllib.request
from datetime import datetime
import selenium as se
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains

base_url = "https://www.luckybrand.com"
base_path = r"E:\\Jelmer\\Uni\\Thesis\\Data\\LUCKYBRAND\\"
section_url_dict_w_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\section_dict_w_LB.p'
section_url_dict_m_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\section_dict_m_LB.p'
categories_m_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\categories_m_LB.p'
categories_w_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\categories_w_LB.p'
scraped_urls_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\scraped_urls_LB.p'


def start_driver(url):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    # UserProfile = "C:\\Users\\" + 's159655' + "\\AppData\\Local\\Google\\Chrome\\User Data\\Default"
    options = se.webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ['enable-automation'])
    # options.add_argument("user-data-dir={}".format(UserProfile))
    options.add_argument('--start-maximized')
    options.add_argument("--incognito")

    chrome_driver = se.webdriver.Chrome(
        executable_path=r"C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\chromedriver_win32\\chromedriver.exe",
        options=options)
    chrome_driver.implicitly_wait(2.5)
    chrome_driver.get(url)
    time.sleep(3)
    chrome_driver.find_element_by_id("gdpr-cookie-accept").click()

    return chrome_driver


def scrape_section(chrome_driver, img_directory, annos_directory):
    """Function which scrapes a certain section of clothing items. Parameters:
    driver: the driver with the section to scrape
    img_dir: a path to save images it finds to
    annos_dir: a path to save annotation files to
    :param annos_directory: directory to descriptions
    :param img_directory: directory to image
    :type chrome_driver: webdriver"""

    # Scroll all the way down to load all images.
    infinite_scroll(chrome_driver)

    section_url = chrome_driver.current_url
    new_items = 0
    print(section_url)
    # Get links to items
    item_links = get_item_links(chrome_driver)
    print("found {} items".format(str(len(item_links))))
    iterator = 0
    for item_link in item_links:
        iterator += 1
        if item_link in scraped_urls:
            continue

        if iterator % 10 == 0:
            print("Scraping item {} of {}...".format(str(iterator), str(len(item_links))))
        chrome_driver.get(item_link)
        try:
            color_urls = get_color_links(chrome_driver)
        except NoSuchElementException:
            color_urls = []

        for color in color_urls:
            if len(color_urls) > 1:
                chrome_driver.get(color)
            if color in scraped_urls:
                continue
            # Try scraping item
            scrape_item(chrome_driver, img_directory, annos_directory)
            scraped_urls.append(chrome_driver.current_url)
            new_items += 1
        if new_items % 10 == 0:
            with open(scraped_urls_file, 'wb') as sc_file:
                pickle.dump(scraped_urls, sc_file)

    with open(scraped_urls_file, 'wb') as sc_file:
        pickle.dump(scraped_urls, sc_file)

    print("Scraped %d new items" % new_items)
    return chrome_driver


def get_item_links(chrome_driver):
    item_links = []
    item_boxes = chrome_driver.find_elements_by_class_name("thumb-link")
    for item_b in item_boxes:
        item_link_full = item_b.get_attribute("href")
        item_link_base = item_link_full.split("?")[0]
        try:
            color_full = "?color=" + str(item_link_full.split("color=")[1][:3])
        except IndexError:
            color_full = ""
        item_links.append(item_link_base + color_full)
    return item_links


def infinite_scroll(chrome_driver):
    """

    :type chrome_driver: webdriver
    """
    # Scroll down
    nr_products = 0
    while True:
        time.sleep(3)
        all_products = chrome_driver.find_elements_by_class_name("thumb-link")
        try:
            description_box = chrome_driver.find_element_by_class_name("sidebar-box")
            loc = description_box.location
            size = description_box.size
            chrome_driver.execute_script("window.scrollTo(0, {});".format(str(loc['y'] - size['height'] - 100)))
        except NoSuchElementException:
            print("can't scroll down to description box as it does not exist")

        nr_products_new = len(all_products)

        if nr_products_new == nr_products:
            break
        else:
            nr_products = nr_products_new
    return chrome_driver


def get_color_links(chrome_driver):
    base = chrome_driver.current_url.split("?")[0]
    color_div = chrome_driver.find_element_by_class_name('swatches.color')
    colors = color_div.find_elements_by_class_name("swatchanchor")
    color_codes = [color.get_attribute("data-lgimg").split("?$hi")[0][-3:] for color in colors if
                   color.get_attribute("data-lgimg") is not None]
    color_links = [base + "?color=" + code for code in color_codes]
    return color_links


def scrape_item(chrome_driver, img_directory, annos_directory):
    """do scraping"""
    item_dict = {}
    date_scraped = str(datetime.date(datetime.now()))
    # BASIC INFORMATION
    # Get page url
    url = chrome_driver.current_url
    current_base_url = url.split("?")[0]
    item_dict["url"] = current_base_url
    item_dict["date_scraped"] = date_scraped

    # Product title
    try:
        product_title = chrome_driver.find_element_by_class_name('product-name').text
        item_dict['product_title'] = product_title
    except NoSuchElementException:
        print("no title found")
        item_dict['product_title'] = None

    # Product color
    item_dict['product_color'] = {}

    # color name
    try:
        product_colorbox = chrome_driver.find_element_by_class_name('attribute.is-color')
        color = product_colorbox.find_element_by_class_name("label").text
        item_dict['product_color']['name'] = color.split(": ")[1]
    except NoSuchElementException:
        print("no color found")
        item_dict['product_color']['name'] = None
    except IndexError:
        item_dict['product_color']['name'] = ""
    # color code
    try:
        color_code = url.split("color=")[1][:3]
    except IndexError:
        print("no color code found")
        item_dict['product_color']['code'] = "NF__"
        color_code = "NF__"
    if color_code == "&dw":
        print("color code not found correctly")
        color_code = "NF__"
    item_dict['product_color']['code'] = color_code

    # Price
    item_dict['price'] = {}
    # Regular price
    try:
        price = chrome_driver.find_element_by_class_name(
            'price-standard').text
        item_dict['price']['regular'] = price
    except NoSuchElementException:
        print("no regular price found")
        item_dict['price']['regular'] = None

    # discounted price
    try:
        price = chrome_driver.find_element_by_class_name('price-sales').text
        item_dict['price']['discount'] = price
    except NoSuchElementException:
        print("no discount price found")
        item_dict['price']['discount'] = None

    # DESCRIPTION AND HIGHLIGHTS
    details_divs = chrome_driver.find_elements_by_class_name('product--info')
    for div in details_divs:
        div_content = div.find_element_by_tag_name("h5").text
        if div_content == "Details":
            # get detail stuff
            try:
                description = div.find_element_by_xpath(".//p").text
                item_dict['description'] = description
            except NoSuchElementException:
                print("No description found")
                item_dict['description'] = None

            # Get highlights
            try:
                highlight_list = div.find_elements_by_tag_name("li")
                highlights = [item.text for item in highlight_list]
                item_dict["highlights"] = highlights
            except NoSuchElementException:
                item_dict["highlights"] = None
                print("no highlights found")

        elif div_content == "Fabric & Care":
            inner_html = div.get_attribute("innerHTML")
            str_to_replace = ["<p>", "<h5>", "<span>", "</p>", "</span>", "</h5>", ","]
            for part in str_to_replace:
                inner_html = inner_html.replace(part, "")
            care_items = inner_html.split("<br>")
            care_item_list = [item for list_part in care_items for item in list_part.split("\n") if
                              item not in ["", '"']]
            item_dict["composition_care_info"] = care_item_list
            # Get fabric & care stuff

    # PRODUCT CODE
    try:
        product_code_div = chrome_driver.find_element_by_class_name('product-number')
        product_code = product_code_div.find_element_by_xpath(".//span/span").text
        item_dict['product_code'] = product_code
    except NoSuchElementException:
        print("Couldnt find the product code")
        item_dict['product_code'] = None

    filename = "LB_" + str(item_dict['product_code']) + "_" + str(item_dict['product_color']['code']).replace("?", "")
    item_dict['filename'] = filename

    # IMAGES
    # Get image links from thumbnails
    thumbnails = chrome_driver.find_elements_by_class_name("productthumbnail")
    img_links = []
    for item in thumbnails:
        base_link = item.get_attribute("src").split("$")[0] + "$hi-res$"
        if base_link not in img_links:
            img_links.append(base_link)
    if not img_links:
        try:
            single_image = chrome_driver.find_element_by_class_name("zoomImg").get_attribute("src")
            img_links = [single_image]
        except NoSuchElementException:
            print("Couldn't find images")
            item_dict["img_urls"] = None
    if img_links:
        item_dict["img_urls"] = img_links
        iterator = 0
        for img_url in img_links:
            urllib.request.urlretrieve(img_url, img_directory + "\\" + filename + "__" + str(iterator) + ".jpg")
            iterator += 1

    with open(annos_directory + "\\" + filename + ".txt", 'w') as item_file:
        json.dump(item_dict, item_file)
    return item_dict


with open(scraped_urls_file, 'rb') as f:
    scraped_urls = pickle.load(f)

with open(section_url_dict_w_file, 'rb') as f:
    section_url_dict_w = pickle.load(f)

with open(section_url_dict_m_file, 'rb') as f:
    section_url_dict_m = pickle.load(f)

with open(categories_m_file, 'rb') as f:
    categories_m = pickle.load(f)

with open(categories_w_file, 'rb') as f:
    categories_w = pickle.load(f)

print(len(scraped_urls))
del(categories_w[13])
driver = start_driver(base_url)

for category in categories_m:
    annos_dir = base_path + "ANNOS\\MEN\\" + category
    img_dir = base_path + "IMG\\MEN\\" + category
    if not os.path.isdir(annos_dir):
        os.mkdir(annos_dir)
        os.mkdir(img_dir)
    driver.get(section_url_dict_m[category])
    driver = scrape_section(driver, img_dir, annos_dir)

for category in categories_w:
    annos_dir = base_path + "ANNOS\\WOMEN\\" + category
    img_dir = base_path + "IMG\\WOMEN\\" + category
    if not os.path.isdir(annos_dir):
        os.mkdir(annos_dir)
        os.mkdir(img_dir)
    driver.get(section_url_dict_w[category])
    driver = scrape_section(driver, img_dir, annos_dir)
