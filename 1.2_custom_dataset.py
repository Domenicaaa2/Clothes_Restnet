import pandas as pd

# create custom dataset similar to the original csv
data = {
    'id': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
    'gender': ['Women', 'Men', 'Women', 'Women', 'Men', 'Women', 'Women', 'Women', 'Women', 'Women', 'Women', 'Men', 'Men', 'Women', 'Women', 'Women', 'Women', 'Women', 'Women', 'Women'],
    'masterCategory': ['Apparel', 'Apparel', 'Apparel', 'Apparel', 'Apparel', 'Accessories', 'Accessories', 'Accessories', 'Accessories', 'Accessories', 'Footwear', 'Footwear', 'Footwear', 'Footwear', 'Footwear', 'Personal Care', 'Personal Care', 'Personal Care', 'Personal Care', 'Personal Care'],
    'subCategory': ['Bottomwear', 'Topwear', 'Topwear', 'Topwear', 'Bottomwear', 'Belts', 'Bags', 'Bags', 'Jewellery', 'Scarves', 'Shoes', 'Flip Flops', 'Shoes', 'Shoes', 'Shoes', 'Lips', 'Fragrance', 'Skin Care', 'Skin Care', 'Skin Care'],
    'articleType': ['Jeans', 'Tshirts', 'Sweatshirts', 'Tshirts', 'Shorts', 'Belts', 'Handbags', 'Handbags', 'Pendant', 'Scarves', 'Casual Shoes', 'Flip Flops', 'Sports Shoes', 'Heels', 'Flats', 'Lipstick', 'Perfume', 'Wrist Bands', 'Headband', 'Cotton pads'],
    'baseColour': ['Blue', 'Brown', 'Red', 'Grey', 'Grey', 'Brown', 'Brown', 'Black', 'Gold', 'Brown', 'Beige', 'Black', 'White', 'Black', 'Black', 'Brown', 'White', 'Pink', 'Pink', 'White'],
    'season': ['Winter', 'Summer', 'Winter', 'Summer', 'Summer', 'All Season', 'All Season', 'All Season', 'All Season', 'Winter', 'All Season', 'Summer', 'All Season', 'Winter', 'All Season', 'All Season', 'All Season', 'All Season', 'All Season', 'All Season'],
    'year': [2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024],
    'usage': ['Casual', 'Casual', 'Sports', 'Casual', 'Casual', 'Casual', 'Party', 'Party', 'Party', 'Casual', 'Casual', 'Casual', 'Sports', 'Party', 'Casual', 'Personal', 'Personal', 'Personal', 'Personal', 'Personal'],
    'productDisplayName': ['Blue Jeans', 'Brown T-shirt', 'Red Sweatshirt', 'Grey T-shirt', 'Grey Shorts', 'Brown Belt', 'Brown Handbag', 'Black Handbag', 'Gold Pendant', 'Brown Scarf', 'Beige Casual Shoes', 'Black Flip Flops', 'White Sports Shoes', 'Black Heels', 'Black Flats', 'Brown Lipstick', 'White Perfume', 'Pink Wrist Bands', 'Pink Headband', 'White Cotton Pads']
}

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
csv_file_path = 'custom_clothing_metadata.csv'
df.to_csv(csv_file_path, index=False)

csv_file_path
