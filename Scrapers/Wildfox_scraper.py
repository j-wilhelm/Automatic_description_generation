import json
import os
import pickle
import urllib.request
from datetime import datetime
import selenium as se
import time
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, TimeoutException, \
    ElementNotInteractableException
from selenium.webdriver import ActionChains

base_url = "https://www.wildfox.com/"
base_path = r"E:\\Jelmer\\Uni\\Thesis\\Data\\WILDFOX\\"
scraped_urls_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\scraped_urls_WILDFOX.p'
categories_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\categories_WILDFOX.p'
section_urls_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\section_urls_WILDFOX.p'


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

    chrome_driver.get(url)
    chrome_driver.implicitly_wait(2.5)
    time.sleep(3)
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

        color_urls = get_color_links(chrome_driver)

        for color in color_urls:
            if color in scraped_urls:
                continue
            if color != driver.current_url:
                chrome_driver.get(color)
            # Try scraping item
            try:
                scrape_item(chrome_driver, img_directory, annos_directory)
                scraped_urls.append(chrome_driver.current_url)
            except TimeoutException:
                print("timeoutexception, restarting driver")
                chrome_driver.close()
                chrome_driver = start_driver(base_url)
        if new_items % 10 == 0:
            with open(scraped_urls_file, 'wb') as sc_file:
                pickle.dump(scraped_urls, sc_file)

    with open(scraped_urls_file, 'wb') as sc_file:
        pickle.dump(scraped_urls, sc_file)

    print("Scraped %d new items" % new_items)
    return chrome_driver


def get_item_links(chrome_driver):
    unique_links = []
    all_products = chrome_driver.find_elements_by_class_name('product_card')
    all_links = [item.get_attribute("href") for item in all_products]

    # Remove duplicates
    for item in all_links:
        if item not in unique_links:
            unique_links.append(item)

    return unique_links


def get_color_links(chrome_driver):
    color_urls = [chrome_driver.current_url]

    additional_colors = chrome_driver.find_elements_by_class_name('swatch-container')

    for color in additional_colors:
        try:
            label_box = color.find_element_by_tag_name("label")
            window_link = label_box.get_attribute("onclick")
            if window_link is not None:
                window_link = window_link.replace("'", "").split(" = ")[1]
                color_urls.append(window_link)
        except NoSuchElementException:
            print("couldn't get color label")
            continue
    return color_urls


def infinite_scroll(chrome_driver):
    """

    :type chrome_driver: webdriver
    """
    # Scroll down
    while True:
        try:
            load_more_button = chrome_driver.find_element_by_id("loadMore")
            chrome_driver.execute_script(
                "window.scrollTo(0, {});".format(load_more_button.location['y'] - load_more_button.size['height'] + 50))
            load_more_button.click()
        except (NoSuchElementException, ElementNotInteractableException):
            return chrome_driver
        except ElementClickInterceptedException:
            try:
                load_more_button = chrome_driver.find_element_by_id("loadMore")
                print("couldn't click load more button, trying again")
                chrome_driver.execute_script("window.scrollTo(0, {});".format(
                    load_more_button.location['y'] - load_more_button.size['height'] - 100))
                load_more_button.click()
            except ElementClickInterceptedException:
                print("still can't click the button, continuing...")
                break

    return chrome_driver


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
        product_title = chrome_driver.find_element_by_class_name(
            'product-title.dn.db-l.f3-ns').find_element_by_tag_name("h1").text
        item_dict['product_title'] = product_title
    except NoSuchElementException:
        print("no title found")
        item_dict['product_title'] = None

    # Product color
    item_dict['product_color'] = {}

    # color name
    try:
        product_colorname = chrome_driver.find_element_by_class_name('normal.tc.tl-l.f7.mb0.mt3-l').text
        item_dict['product_color']['name'] = product_colorname
    except NoSuchElementException:
        print("no color found")
        item_dict['product_color']['name'] = None
    # color code
    item_dict['product_color']['code'] = None

    # Price
    try:
        price = chrome_driver.find_element_by_class_name('money.f5.fw8.db.tc').get_attribute("innerHTML")
        item_dict['price'] = price
    except NoSuchElementException:
        print("no regular price found")
        item_dict['price'] = None

    # Product code
    try:
        product_code = chrome_driver.find_element_by_class_name('style_no.avantgarde.fw100.f5.silver').text
        item_dict['product_code'] = product_code.split(" ")[-1]
    except NoSuchElementException:
        print("no regular product_code found")
        item_dict['product_code'] = None

    # DESCRIPTION AND HIGHLIGHTS
    try:
        details_div = chrome_driver.find_element_by_class_name('description')
        # description:
        try:
            description = details_div.find_element_by_xpath(".//p[1]").text
            item_dict['description'] = description
        except NoSuchElementException:
            print("no description found...")
            item_dict['description'] = None

        # Highlights
        highlights = []
        try:
            par = 3
            while True:
                highlight_item = details_div.find_element_by_xpath(".//p[{}]".format(str(par))).text
                if highlight_item == "Details":
                    break
                highlights.append(highlight_item)
                par += 1

            item_dict['highlights'] = highlights
        except NoSuchElementException:
            print("something went wrong while finding highlights...")
            item_dict['highlights'] = None
    except NoSuchElementException:
        print("no description / highlights box found")
        item_dict['description'] = None
        item_dict['highlights'] = None

    filename = "WF_" + str(item_dict['product_code']) + "_" + str(item_dict['product_color']['name']).replace("?", "").replace("/","")
    item_dict['filename'] = filename

    # IMAGES
    # get first image link

    # click through thumbnails
    iterator = 0

    product_images = chrome_driver.find_elements_by_class_name("product-single__thumbnail--product-template")
    img_links = [img.get_attribute("href") for img in product_images]
    if not img_links:
        print("No images found")
    for img_url in img_links:
        urllib.request.urlretrieve(img_url, img_directory + "\\" + filename + "__" + str(iterator) + ".jpg")
        iterator += 1
    item_dict["img_urls"] = img_links

    with open(annos_directory + "\\" + filename + ".txt", 'w') as item_file:
        json.dump(item_dict, item_file)
    return item_dict


# SCRAPING
with open(section_urls_file, 'rb') as f:
    section_url_dict = pickle.load(f)

with open(categories_file, 'rb') as f:
    categories = pickle.load(f)

with open(scraped_urls_file, 'rb') as f:
    scraped_urls = pickle.load(f)

driver = start_driver(base_url)
for section in categories:
    driver.get(section_url_dict[section])
    img_dir = base_path + "IMG\\" + section
    annos_dir = base_path + "ANNOS\\" + section
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
        os.mkdir(annos_dir)

    scrape_section(driver, img_dir, annos_dir)
driver.close()