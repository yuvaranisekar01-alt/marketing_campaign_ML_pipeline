numerical_cols = ['Duration', 'Impressions', 'Clicks', 'Leads', 'Conversions',
       'Acquisition_Cost', 'ROI', 'Engagement_Score', 'Month', 'Year',
       'Quarter', 'CTR', 'Conversion_rate', 'Lead_Rate', 'Revenue_per_lead',
       'Cost_per_click']

categorical_cols = ['Campaign_Type', 'Target_Audience', 'Channel_Used', 'Language',
       'Customer_Segment', 'Brand']


target_reg = 'Revenue'
target_cls = 'Profit_loss'


selected_features = ['Cost_per_click','Acquisition_Cost', 'Conversion_rate',
                 'Campaign_Type', 'Channel_Used', 'Brand']

selected_cat_cols = ['Campaign_Type', 'Channel_Used', 'Brand']
selected_num_cols = ['Cost_per_click','Acquisition_Cost', 'Conversion_rate']

