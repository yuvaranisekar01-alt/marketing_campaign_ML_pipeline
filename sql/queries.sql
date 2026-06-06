--Table 1

CREATE TABLE IF NOT EXISTS raw_campaigns (
    id                  SERIAL PRIMARY KEY,
    campaign_type       VARCHAR(50),
    target_audience     VARCHAR(50),
    channel_used        VARCHAR(100),
    language            VARCHAR(30),
    customer_segment    VARCHAR(50),
    brand               VARCHAR(30),
    duration            FLOAT,
    impressions         FLOAT,
    clicks              FLOAT,
    leads               FLOAT,
    conversions         FLOAT,
    revenue             FLOAT,
    acquisition_cost    FLOAT,
    roi                 FLOAT,
    engagement_score    FLOAT,
    date                DATE
);

-- Table 2

CREATE TABLE IF NOT EXISTS feature_engineered_campaigns (
    id                  SERIAL PRIMARY KEY,
    campaign_type       VARCHAR(50),
    channel_used        VARCHAR(100),
    brand               VARCHAR(30),
    conversion_rate     FLOAT,
    acquisition_cost    FLOAT,
    revenue_per_lead    FLOAT,
    cost_per_click      FLOAT,
    roi                 FLOAT,
    month               INT,
    year                INT,
    quarter             INT,
    revenue             FLOAT,
    profit_loss         VARCHAR(10)
);

-- Table 3

CREATE TABLE IF NOT EXISTS model_predictions (
    id                  SERIAL PRIMARY KEY,
    brand               VARCHAR(30),
    campaign_type       VARCHAR(50),
    channel_used        VARCHAR(100),
    acquisition_cost    FLOAT,
    predicted_revenue   FLOAT,
    predicted_profit_loss VARCHAR(10),
    prediction_date     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Analysis Queries

-- total revenue by brand
select brand ,
    round(sum(revenue):: numeric,2) as total_revenue,
    round(avg(revenue):: numeric, 2) as avg_revenue,
    count(*) as total_campaign
from feature_engineered_campaigns
group by brand
order by total_revenue desc;

-- profit vs loss count by brand
select brand, profit_loss,
    count(*) as total_campaign,
    round(avg(revenue):: numeric,2) as avg_revenue
from feature_engineered_campaigns
group by brand , profit_loss
order by brand, profit_loss;

-- best performing campaign type
select campaign_type, 
    round(avg(revenue):: numeric, 2) as avg_revenue,
    round(avg(conversion_rate):: numeric, 4) as avg_conversion_rate,
    round(avg(acquisition_cost):: numeric, 4) as  avg_cost,
    count(*) as total_campaign
from feature_engineered_campaigns
group by campaign_type
order by avg_revenue desc;

-- monthly revenue trend
select year, month,
    round(sum(revenue):: numeric, 2) as total_revenue,
    round(avg(acquisition_cost):: numeric, 2) as avg_cost,
    count(*) as total_campaign
from feature_engineered_campaigns
group by year, month
order by year, month;

-- top 10 highest revenue campaigns
select brand, campaign_type, channel_used,
    revenue, acquisition_cost, profit_loss
from feature_engineered_campaigns
order by revenue desc
limit 10;

-- profit loss ratio by channel
select channel_used,
    count(*) filter(where profit_loss = 'Profit') as profit_count,
    count(*) filter(where profit_loss = 'Loss') as loss_count,
    round(
        count(*) filter(where profit_loss = 'Profit') * 100 / count(*) , 2 ) as profit_percentage
from feature_engineered_campaigns
group by channel_used
order by profit_percentage desc;

-- average cost per click by brand
select brand,
    round(avg(cost_per_click):: numeric,2) as avg_cost_per_click,
    round(avg(conversion_rate):: numeric, 4) as avg_conversion_rate
from feature_engineered_campaigns
group by brand
order by avg_cost_per_click;

-- quarterly performance
select year, quarter,
    round(sum(revenue):: numeric, 2) as total_revenue,
    round(avg(acquisition_cost):: numeric, 4) as avg_acquisition_cost,
    count(*) as total_campaign
from feature_engineered_campaigns
group by year, quarter
order by year, quarter;











