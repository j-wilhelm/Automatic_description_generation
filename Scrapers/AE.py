import json
import os
import pickle
import time
import urllib.request
from datetime import datetime
from urllib.error import URLError
import selenium as se
from selenium.common.exceptions import NoSuchElementException, \
    StaleElementReferenceException, ElementClickInterceptedException, TimeoutException
from selenium.webdriver import ActionChains

base_url = "https://www.ae.com/us/en"
base_path = r"E:\\Jelmer\\Uni\\Thesis\\Data\\AE\\"
section_url_dict_w_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\section_dict_w_AE.p'
section_url_dict_m_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\section_dict_m_AE.p'
scraped_urls_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\scraped_urls_AE.p'
bad_urls_file = r'C:\\Users\\s159655\\Documents\\JADS\\Thesis\\Code\\Scrapers\\Bat_files\\bad_urls_AE.p'


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
    chrome_driver.implicitly_wait(5)
    chrome_driver.get(url)

    # Accept privacy message
    privacy_button = chrome_driver.find_element_by_id("ember1375")
    privacy_button.click()

    # stay on US store
    current_store_button = chrome_driver.find_element_by_class_name(
        "flag-wrapper.clickable.ember-view.country-current.qa-country-current")
    current_store_button.click()

    # close the newspaper part as well
    time.sleep(2.5)
    try:
        signup_close = chrome_driver.find_element_by_class_name("close.clickable.btn-cancel.qa-btn-cancel")
        signup_close.click()
    except NoSuchElementException:
        print("couldn't find close button \nwait a few seconds before trying again...")
        time.sleep(2.5)
        try:
            signup_close = chrome_driver.find_element_by_class_name("close.clickable.btn-cancel.qa-btn-cancel")
            signup_close.click()
        except ElementClickInterceptedException:
            signup_loc = signup_close.location
            chrome_driver.execute_script("window.scrollTo({},{});".format(signup_loc['x'], signup_loc['y']))
            signup_close.click()
        except NoSuchElementException:
            print("couldn't find close button.")
    except ElementClickInterceptedException:
        signup_loc = signup_close.location
        chrome_driver.execute_script("window.scrollTo({},{});".format(signup_loc['x'], signup_loc['y']))
        signup_close.click()
    chrome_driver.implicitly_wait(1)
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
        color_urls = []
        if item_link in scraped_urls:
            continue

        if iterator % 10 == 0:
            print("Scraping item {} of {}...".format(str(iterator), str(len(item_links))))
        chrome_driver.get(item_link)

        color_boxes = chrome_driver.find_elements_by_class_name("swatch")
        current_base_url = chrome_driver.current_url.split("?")[0][:-13]
        for color_box in color_boxes:
            product_code = color_box.get_attribute("data-proddata-key")
            next_color_url = current_base_url + product_code
            new_items += 1
            color_urls.append(next_color_url)

        for color in color_urls:
            print(color)
            chrome_driver.get(color)
            if color in scraped_urls:
                continue
            # Try scraping item
            try:
                scrape_item(chrome_driver, img_directory, annos_directory)
                scraped_urls.append(chrome_driver.current_url)
            except TimeoutException:
                print("timeoutexception, restarting driver")
                chrome_driver.close()
                chrome_driver = start_driver(current_base_url)
        if new_items % 10 == 0:
            with open(scraped_urls_file, 'wb') as sc_file:
                pickle.dump(scraped_urls, sc_file)

    with open(scraped_urls_file, 'wb') as sc_file:
        pickle.dump(scraped_urls, sc_file)

    with open(bad_urls_file, 'wb') as sc_file:
        pickle.dump(bad_urls, sc_file)
    print("Scraped %d new items" % new_items)
    return chrome_driver


def get_item_links(chrome_driver):
    item_links = []
    item_boxes = chrome_driver.find_elements_by_class_name("xm-link-to.qa-xm-link-to.tile-link")
    for item_b in item_boxes:
        item_link_full = item_b.get_attribute("href")
        item_link_base = item_link_full.split("?")[0]
        item_links.append(item_link_base)
    return item_links


def infinite_scroll(chrome_driver):
    """

    :type chrome_driver: webdriver
    """
    # Scroll down
    nr_products = 0
    while True:
        all_products = chrome_driver.find_elements_by_class_name("xm-link-to.qa-xm-link-to.tile-link")
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(0.1)
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.5);")
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.25);")
        time.sleep(0.1)
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.1);")
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.05);")
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.025);")
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.001);")
        time.sleep(3)
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.5);")
        time.sleep(0.1)
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.25);")
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.1);")
        time.sleep(0.1)
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.05);")
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.025);")
        time.sleep(0.1)
        chrome_driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.011);")
        nr_products_new = len(all_products)

        if nr_products_new == nr_products:
            break
        else:
            nr_products = nr_products_new
    print(nr_products)
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
        product_title = chrome_driver.find_element_by_class_name('product-name').text
        item_dict['product_title'] = product_title
    except NoSuchElementException:
        print("no title found")
        item_dict['product_title'] = None

    # Product color
    item_dict['product_color'] = {}

    # color name
    try:
        product_colorname = chrome_driver.find_element_by_class_name('psp-product-txt.psp-product-color').text
        item_dict['product_color']['name'] = product_colorname
    except NoSuchElementException:
        print("no color found")
        item_dict['product_color']['name'] = None
    # color code
    try:
        product_colorcode = chrome_driver.find_element_by_class_name('equity-item-id.equity-item-color-id').text
        item_dict['product_color']['code'] = product_colorcode
    except NoSuchElementException:
        print("no color found")
        item_dict['product_color']['code'] = None

    # Price
    item_dict['price'] = {}
    # Regular price
    try:
        price = chrome_driver.find_element_by_class_name(
            'product-list-price.product-list-price-on-sale.ember-view').text
        item_dict['price']['regular'] = price
    except NoSuchElementException:
        print("no regular price found")
        item_dict['price']['regular'] = None

    # discounted price
    try:
        price = chrome_driver.find_element_by_class_name('product-sale-price.ember-view').text
        item_dict['price']['discount'] = price
    except NoSuchElementException:
        print("no discount price found")
        item_dict['price']['discount'] = None

    # DESCRIPTION AND HIGHLIGHTS
    try:
        details_div = chrome_driver.find_element_by_class_name('equity-group-item.equity-group-item-details')
        # description:
        try:
            description = details_div.find_element_by_class_name("equity-group-intro-txt.equity-group-intro-equit").text
            item_dict['description'] = description
        except NoSuchElementException:
            print("no description found...")
            item_dict['description'] = None

        # Highlights
        try:
            highlight_items = details_div.find_elements_by_class_name("equity-group-list-item")
            highlights = [highlight.text for highlight in highlight_items]
            item_dict['highlights'] = highlights
        except NoSuchElementException:
            print("no highlights found...")
            item_dict['highlights'] = None
    except NoSuchElementException:
        print("no description / highlights box found")
        item_dict['description'] = None
        item_dict['highlights'] = None

    # PRODUCT CODE
    try:
        product_code = chrome_driver.find_element_by_class_name('equity-item-id.equity-item-prod-id').text
        item_dict['product_code'] = product_code
    except NoSuchElementException:
        print("Couldnt find the product code")
        item_dict['product_code'] = None

    filename = "AE_" + str(item_dict['product_code']) + "_" + str(item_dict['product_color']['code']).replace("?", "")
    item_dict['filename'] = filename

    # Get some material / care info
    try:
        material_box = chrome_driver.find_element_by_class_name("equity-group-item.equity-group-item-material")
        care_list = [care_item.text for care_item in material_box.find_elements_by_class_name("equity-group-list-item")]
        item_dict['composition_care_info'] = care_list
    except NoSuchElementException:
        print("No composition / care info found")
        item_dict["composition_care_info"] = None

    # IMAGES
    # get first image link

    # click through thumbnails
    img_links = []
    image_thumbnails = []
    iterator = 0
    attempts = 0
    while attempts < 3:
        try:
            thumbnail_list = chrome_driver.find_element_by_class_name(
                "carousel-thumbs.carousel-indicators.__52d08.ember-view")
            image_thumbnails = thumbnail_list.find_elements_by_class_name("img-responsive")
            break
        except StaleElementReferenceException:
            attempts += 1
            continue

        except NoSuchElementException:
            print("no additional images found")
            try:
                img_url = chrome_driver.find_element_by_class_name(
                    "zooming-image.qa-zooming-image.__d95dc.image.lazyload").get_attribute("src")
                urllib.request.urlretrieve(img_url, img_directory + "\\" + filename + "__" + str(0) + ".jpg")
            except NoSuchElementException:
                print("No images found")
            except URLError:
                print("URLError, item appended to bad_urls")
                bad_urls.append((filename + "__" + str(iterator), img_directory, img_url))
            break

    print("Found %d images for this item." % len(image_thumbnails))
    if image_thumbnails:
        for img_button in image_thumbnails:
            try:
                if img_button.get_attribute("alt") == "Logos":
                    continue
                chrome_driver.execute_script(
                    "window.scrollTo({},{});".format(0, img_button.location['y'] - img_button.size['height']))
                img_button.click()
                img_url = chrome_driver.find_element_by_class_name(
                    "zooming-image.qa-zooming-image.__d95dc.image.lazyload").get_attribute("src")
                img_links.append(img_url)
                urllib.request.urlretrieve(img_url, img_directory + "\\" + filename + "__" + str(iterator) + ".jpg")
            except URLError as e:
                print(str(e))
                time.sleep(2)
                try:
                    urllib.request.urlretrieve(img_url, img_directory + "\\" + filename + "__" + str(iterator) + ".jpg")
                except URLError:
                    print("Still a URLError, item appended to bad_urls")
                    bad_urls.append((filename + "__" + str(iterator), img_directory, img_url))
            except StaleElementReferenceException as e:
                print(str(e))
                continue
            except ElementClickInterceptedException as e:
                print(str(e))
                try:
                    time.sleep(2.5)
                    chrome_driver.execute_script(
                        "window.scrollTo({},{});".format(0, img_button.location['y'] - img_button.size['height'] - 50))
                    img_button.click()
                    img_url = chrome_driver.find_element_by_class_name(
                        "zooming-image.qa-zooming-image.__d95dc.image.lazyload").get_attribute("src")
                    img_links.append(img_url)
                except ElementClickInterceptedException:
                    print("still cant click on thumbnail due to interception, skip this picture")
                    continue
                except StaleElementReferenceException:
                    print("Stale element reference exception when trying to click this image thumbnail")
                    continue
            iterator += 1
        item_dict["img_urls"] = img_links
    else:
        print("Couldn't find any image links")

    with open(annos_directory + "\\" + filename + ".txt", 'w') as item_file:
        json.dump(item_dict, item_file)
    return item_dict


with open(scraped_urls_file, 'rb') as f:
    scraped_urls = pickle.load(f)

with open(section_url_dict_w_file, 'rb') as f:
    section_url_dict_wo = pickle.load(f)

with open(section_url_dict_m_file, 'rb') as f:
    section_url_dict_ma = pickle.load(f)

with open(bad_urls_file, 'rb') as f:
    bad_urls = pickle.load(f)

print(len(scraped_urls))

driver = start_driver(base_url)

categories_ma = ["Tops", "Jeans", "Bottoms", "Jackets & Coats", "Underwear", "Loungewear", "Shoes",
                 "Accessories & Socks", "Cologne & Grooming", "Gifts & Tech", "College", "NBA", "MLB", "NFL"]
for category in categories_ma:
    annos_dir = base_path + "ANNOS\\MEN\\" + category
    img_dir = base_path + "IMG\\MEN\\" + category
    if not os.path.isdir(annos_dir):
        os.mkdir(annos_dir)
        os.mkdir(img_dir)
    driver.get(section_url_dict_ma[category])
    driver = scrape_section(driver, img_dir, annos_dir)

categories_wo = ["Tops", "Jeans", "Bottoms", "Jackets & Coats", "Dresses", "Loungewear", "Accessories & Socks", "Shoes",
                 "Swimsuits", "Gifts & Tech", "Perfume & Beauty", "Bras", "Bralettes", "Undies", "Leggings"]

for category in categories_wo:
    annos_dir = base_path + "ANNOS\\WOMEN\\" + category
    img_dir = base_path + "IMG\\WOMEN\\" + category
    if not os.path.isdir(annos_dir):
        os.mkdir(annos_dir)
        os.mkdir(img_dir)
    driver.get(section_url_dict_wo[category])
    driver = scrape_section(driver, img_dir, annos_dir)
