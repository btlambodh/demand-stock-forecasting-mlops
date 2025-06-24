#!/usr/bin/env python3
"""
Dashboard Data Viewer: Simple web-based dashboard for 
Chinese Produce Market Analytics, updated to handle 
correct column names and data structure

Author: Bhupal Lambodhar
Email: btiduwarlambodhar@sandiego.edu
"""

import json
import os
import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DashboardViewer:
    """Dashboard data viewer using Streamlit - Updated Version"""
    
    def __init__(self, data_path: str = 'dashboard_data'):
        self.data_path = data_path
        self.dashboard_data = {}
        self.load_dashboard_data()
    
    def load_dashboard_data(self):
        """Load dashboard data from JSON files"""
        json_file = os.path.join(self.data_path, 'dashboard_data.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                self.dashboard_data = json.load(f)
    
    def show_overview(self):
        """Display overview metrics"""
        st.header("Market Overview")
        
        if 'overview' in self.dashboard_data:
            overview = self.dashboard_data['overview']
            
            # Create metrics columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'total_records' in overview:
                    st.metric("Total Records", f"{overview['total_records']['total_records']:,}")
            
            with col2:
                if 'unique_items' in overview:
                    st.metric("Unique Categories", f"{overview['unique_items']['unique_items']:,}")
                elif 'unique_categories' in overview:
                    st.metric("Unique Categories", f"{overview['unique_categories']['unique_categories']:,}")
            
            with col3:
                if 'total_revenue' in overview:
                    revenue = overview['total_revenue']['total_revenue']
                    st.metric("Total Revenue", f"{revenue:,.0f} RMB")
            
            with col4:
                if 'total_revenue' in overview:
                    avg_revenue = overview['total_revenue']['avg_revenue']
                    st.metric("Avg Revenue", f"{avg_revenue:.2f} RMB")
            
            # Additional metrics row
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                if 'total_revenue' in overview:
                    transaction_count = overview['total_revenue']['transaction_count']
                    st.metric("Total Transactions", f"{transaction_count:,}")
            
            # Data period
            if 'date_range' in overview:
                date_range = overview['date_range']
                st.info(f"Data Period: {date_range['earliest_date']} to {date_range['latest_date']}")
        else:
            st.warning("Overview data not available. Please generate dashboard data first.")
    
    def show_revenue_trends(self):
        """Display revenue trends"""
        st.header("Revenue Trends")
        
        if 'revenue_trends' in self.dashboard_data:
            trends = self.dashboard_data['revenue_trends']
            
            # Monthly revenue trend
            if 'monthly_revenue' in trends:
                st.subheader("Monthly Revenue Trend")
                df = pd.DataFrame(trends['monthly_revenue'])
                
                if not df.empty:
                    # Create date column for plotting
                    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
                    
                    fig = px.line(df, x='date', y='monthly_revenue', 
                                 title='Monthly Revenue Trend',
                                 labels={'monthly_revenue': 'Revenue (RMB)', 'date': 'Date'})
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show recent data
                    st.subheader("Recent Monthly Performance")
                    recent_data = df.tail(6)[['year', 'month', 'monthly_revenue', 'avg_price', 'transaction_count']]
                    st.dataframe(recent_data, use_container_width=True)
                else:
                    st.warning("No monthly revenue data available")
            
            # Seasonal analysis
            if 'seasonal_revenue' in trends:
                st.subheader("Seasonal Performance")
                df_seasonal = pd.DataFrame(trends['seasonal_revenue'])
                
                if not df_seasonal.empty:
                    fig = px.bar(df_seasonal, x='Season', y='seasonal_revenue',
                                title='Revenue by Season',
                                labels={'seasonal_revenue': 'Revenue (RMB)'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Season details
                    st.dataframe(df_seasonal, use_container_width=True)
                else:
                    st.warning("No seasonal data available")
            
            # Daily trends (if available)
            if 'daily_revenue_trend' in trends:
                st.subheader("Recent Daily Trends (Last 90 Days)")
                df_daily = pd.DataFrame(trends['daily_revenue_trend'])
                
                if not df_daily.empty:
                    df_daily['date'] = pd.to_datetime(df_daily['date'])
                    fig = px.line(df_daily, x='date', y='daily_revenue',
                                 title='Daily Revenue Trend',
                                 labels={'daily_revenue': 'Revenue (RMB)', 'date': 'Date'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Revenue trends data not available. Please generate dashboard data first.")
    
    def show_category_analysis(self):
        """Display category analysis"""
        st.header("Category Analysis")
        
        if 'category_analysis' in self.dashboard_data:
            analysis = self.dashboard_data['category_analysis']
            
            # Top categories
            if 'top_categories_by_revenue' in analysis:
                st.subheader("Top Categories by Revenue")
                df = pd.DataFrame(analysis['top_categories_by_revenue'])
                
                if not df.empty:
                    # Bar chart
                    fig = px.bar(df.head(10), x='Category_Name', y='total_revenue',
                                title='Top 10 Categories by Revenue',
                                labels={'total_revenue': 'Revenue (RMB)', 'Category_Name': 'Category'})
                    fig.update_xaxis(tickangle=45)
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    st.subheader("Category Performance Details")
                    display_df = df[['Category_Name', 'total_revenue', 'avg_price', 'total_quantity', 'transaction_count']]
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("No category revenue data available")
            
            # Category growth over years
            if 'category_growth' in analysis:
                st.subheader("Category Growth Over Years")
                df_growth = pd.DataFrame(analysis['category_growth'])
                
                if not df_growth.empty:
                    # Get top 5 categories for growth visualization
                    top_categories = df_growth.groupby('Category_Name')['yearly_revenue'].sum().nlargest(5).index
                    df_top_growth = df_growth[df_growth['Category_Name'].isin(top_categories)]
                    
                    fig = px.line(df_top_growth, x='year', y='yearly_revenue', color='Category_Name',
                                 title='Top 5 Categories - Yearly Revenue Growth',
                                 labels={'yearly_revenue': 'Revenue (RMB)', 'year': 'Year'})
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Category seasonality
            if 'category_seasonality' in analysis:
                st.subheader("Category Seasonality Heatmap")
                df_season = pd.DataFrame(analysis['category_seasonality'])
                
                if not df_season.empty:
                    # Create pivot table for heatmap
                    pivot_df = df_season.pivot(index='Category_Name', columns='Season', values='seasonal_revenue')
                    pivot_df = pivot_df.fillna(0)
                    
                    fig = px.imshow(pivot_df, 
                                   title='Category Performance by Season (Revenue)',
                                   labels={'color': 'Revenue (RMB)'},
                                   aspect='auto')
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Category analysis data not available. Please generate dashboard data first.")
    
    def show_price_analysis(self):
        """Display price analysis"""
        st.header("Price Analysis")
        
        if 'price_analysis' in self.dashboard_data:
            analysis = self.dashboard_data['price_analysis']
            
            # Price distribution
            if 'price_distribution' in analysis:
                st.subheader("Price Distribution")
                df = pd.DataFrame(analysis['price_distribution'])
                
                if not df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(df, values='transaction_count', names='price_range',
                                    title='Transaction Distribution by Price Range')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(df, x='price_range', y='range_revenue',
                                    title='Revenue by Price Range')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(df, use_container_width=True)
            
            # Price trends over time
            if 'price_trends_by_month' in analysis:
                st.subheader("Price Trends Over Time")
                df = pd.DataFrame(analysis['price_trends_by_month'])
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['date'], y=df['avg_monthly_price'],
                                           mode='lines', name='Average Price', line=dict(width=3)))
                    fig.add_trace(go.Scatter(x=df['date'], y=df['min_monthly_price'],
                                           mode='lines', name='Min Price', opacity=0.6))
                    fig.add_trace(go.Scatter(x=df['date'], y=df['max_monthly_price'],
                                           mode='lines', name='Max Price', opacity=0.6))
                    
                    fig.update_layout(title='Price Trends by Month',
                                     xaxis_title='Date',
                                     yaxis_title='Price (RMB)',
                                     height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Most volatile items
            if 'most_volatile_items' in analysis:
                st.subheader("Most Volatile Items")
                df = pd.DataFrame(analysis['most_volatile_items'])
                
                if not df.empty:
                    st.dataframe(df.head(15), use_container_width=True)
                    
                    # Volatility chart
                    fig = px.bar(df.head(10), x='Item_Code', y='avg_volatility',
                                title='Top 10 Most Volatile Items',
                                labels={'avg_volatility': 'Price Volatility'})
                    fig.update_xaxis(tickangle=45)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Price analysis data not available. Please generate dashboard data first.")
    
    def show_market_insights(self):
        """Display market insights"""
        st.header("Market Insights")
        
        if 'market_insights' in self.dashboard_data:
            insights = self.dashboard_data['market_insights']
            
            # Top performing items
            if 'top_performing_items' in insights:
                st.subheader("Top Performing Items")
                df = pd.DataFrame(insights['top_performing_items'])
                
                if not df.empty:
                    st.dataframe(df.head(20), use_container_width=True)
                    
                    # Revenue visualization
                    fig = px.bar(df.head(10), x='Item_Code', y='total_revenue',
                                title='Top 10 Items by Revenue',
                                labels={'total_revenue': 'Revenue (RMB)'})
                    fig.update_xaxis(tickangle=45)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Weekend vs Weekday
            if 'weekend_vs_weekday' in insights:
                st.subheader("Weekend vs Weekday Performance")
                df = pd.DataFrame(insights['weekend_vs_weekday'])
                
                if not df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(df, x='day_type', y='total_revenue',
                                    title='Revenue: Weekend vs Weekday',
                                    color='day_type')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(df, x='day_type', y='avg_price',
                                    title='Average Price: Weekend vs Weekday',
                                    color='day_type')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(df, use_container_width=True)
            
            # Holiday impact
            if 'holiday_impact' in insights:
                st.subheader("Holiday Impact")
                df = pd.DataFrame(insights['holiday_impact'])
                
                if not df.empty:
                    fig = px.bar(df, x='day_type', y='total_revenue',
                                title='Revenue by Day Type',
                                color='day_type')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df, use_container_width=True)
            
            # Loss rate impact
            if 'loss_rate_impact' in insights:
                st.subheader("Loss Rate Impact")
                df = pd.DataFrame(insights['loss_rate_impact'])
                
                if not df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(df, values='item_count', names='loss_category',
                                    title='Items by Loss Rate Category')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(df, x='loss_category', y='total_revenue',
                                    title='Revenue by Loss Rate Category')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(df, use_container_width=True)
        else:
            st.warning("Market insights data not available. Please generate dashboard data first.")
    
    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="Chinese Produce Market Analytics",
            page_icon="chart",
            layout="wide"
        )
        
        st.title("Chinese Produce Market Analytics Dashboard")
        st.markdown("---")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page",
            ["Overview", "Revenue Trends", "Category Analysis", "Price Analysis", "Market Insights"]
        )
        
        # Display selected page
        if page == "Overview":
            self.show_overview()
        elif page == "Revenue Trends":
            self.show_revenue_trends()
        elif page == "Category Analysis":
            self.show_category_analysis()
        elif page == "Price Analysis":
            self.show_price_analysis()
        elif page == "Market Insights":
            self.show_market_insights()
        
        # Data refresh info
        st.sidebar.markdown("---")
        st.sidebar.info(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Data status
        data_status = "Loaded" if self.dashboard_data else "No Data"
        st.sidebar.metric("Data Status", data_status)
        
        if self.dashboard_data:
            data_components = len(self.dashboard_data)
            st.sidebar.metric("Data Components", data_components)
        
        # Download data
        if st.sidebar.button("Download Raw Data"):
            if self.dashboard_data:
                json_data = json.dumps(self.dashboard_data, indent=2, default=str)
                st.sidebar.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"dashboard_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            else:
                st.sidebar.error("No data available to download")


# Enhanced HTML dashboard generator
def generate_html_dashboard(data_path: str = 'dashboard_data'):
    """Generate enhanced HTML dashboard with better styling"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chinese Produce Market Analytics</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 40px;
                color: #333;
            }
            .header h1 {
                font-size: 2.5em;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }
            .metric { 
                padding: 25px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 12px; 
                text-align: center; 
                color: white;
                box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
                transition: transform 0.3s ease;
            }
            .metric:hover {
                transform: translateY(-5px);
            }
            .metric h3 { 
                margin: 0; 
                font-size: 14px; 
                text-transform: uppercase; 
                opacity: 0.9; 
                letter-spacing: 1px;
            }
            .metric p { 
                margin: 10px 0 0 0; 
                font-size: 28px; 
                font-weight: bold; 
            }
            .chart-container { 
                margin: 30px 0; 
                background: #f8f9fa;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .chart { 
                margin: 20px 0; 
            }
            h2 { 
                color: #333; 
                border-bottom: 3px solid #667eea; 
                padding-bottom: 10px; 
                margin-top: 40px;
            }
            .status {
                padding: 20px;
                background: #e8f5e8;
                border-left: 5px solid #4caf50;
                margin: 20px 0;
                border-radius: 8px;
            }
            .footer {
                text-align: center;
                margin-top: 50px;
                padding: 20px;
                color: #666;
                border-top: 1px solid #eee;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Chinese Produce Market Analytics</h1>
                <p>Business Intelligence Dashboard</p>
            </div>
            
            <div class="status">
                <strong>Dashboard Status:</strong> Active and Updated
                <br><strong>Last Generated:</strong> <span id="last-updated">Loading...</span>
            </div>
            
            <h2>Key Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Total Revenue</h3>
                    <p id="total-revenue">Loading...</p>
                </div>
                <div class="metric">
                    <h3>Total Records</h3>
                    <p id="total-records">Loading...</p>
                </div>
                <div class="metric">
                    <h3>Categories</h3>
                    <p id="unique-categories">Loading...</p>
                </div>
                <div class="metric">
                    <h3>Avg Revenue</h3>
                    <p id="avg-revenue">Loading...</p>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Revenue Trends</h2>
                <div id="revenue-chart" class="chart"></div>
            </div>
            
            <div class="chart-container">
                <h2>Top Categories</h2>
                <div id="category-chart" class="chart"></div>
            </div>
            
            <div class="footer">
                <p>Chinese Produce Market Analytics Dashboard © 2024</p>
                <p>Generated for data-driven insights</p>
            </div>
        </div>
        
        <script>
            // Update timestamp
            document.getElementById('last-updated').textContent = new Date().toLocaleString();
            
            // Load and display dashboard data
            fetch('dashboard_data.json')
                .then(response => response.json())
                .then(data => {
                    console.log('Dashboard data loaded:', data);
                    
                    // Update metrics
                    if (data.overview) {
                        if (data.overview.total_revenue) {
                            const revenue = data.overview.total_revenue.total_revenue;
                            const avgRevenue = data.overview.total_revenue.avg_revenue;
                            document.getElementById('total-revenue').textContent = 
                                revenue ? (revenue/1000000).toFixed(1) + 'M RMB' : 'N/A';
                            document.getElementById('avg-revenue').textContent = 
                                avgRevenue ? avgRevenue.toFixed(1) + ' RMB' : 'N/A';
                        }
                        
                        if (data.overview.total_records) {
                            document.getElementById('total-records').textContent = 
                                data.overview.total_records.total_records.toLocaleString();
                        }
                        
                        if (data.overview.unique_items) {
                            document.getElementById('unique-categories').textContent = 
                                data.overview.unique_items.unique_items.toLocaleString();
                        } else if (data.overview.unique_categories) {
                            document.getElementById('unique-categories').textContent = 
                                data.overview.unique_categories.unique_categories.toLocaleString();
                        }
                    }
                    
                    // Revenue trends chart
                    if (data.revenue_trends && data.revenue_trends.monthly_revenue) {
                        const monthlyData = data.revenue_trends.monthly_revenue;
                        const trace = {
                            x: monthlyData.map(d => `${d.year}-${d.month.toString().padStart(2, '0')}`),
                            y: monthlyData.map(d => d.monthly_revenue),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Monthly Revenue',
                            line: {color: '#667eea', width: 3},
                            marker: {color: '#764ba2', size: 8}
                        };
                        
                        Plotly.newPlot('revenue-chart', [trace], {
                            title: 'Monthly Revenue Trend',
                            xaxis: { title: 'Date' },
                            yaxis: { title: 'Revenue (RMB)' },
                            showlegend: false,
                            margin: {t: 40, b: 50, l: 80, r: 30}
                        });
                    }
                    
                    // Category chart
                    if (data.category_analysis && data.category_analysis.top_categories_by_revenue) {
                        const categoryData = data.category_analysis.top_categories_by_revenue.slice(0, 10);
                        const trace = {
                            x: categoryData.map(d => d.Category_Name),
                            y: categoryData.map(d => d.total_revenue),
                            type: 'bar',
                            name: 'Revenue by Category',
                            marker: {
                                color: categoryData.map((d, i) => `hsl(${240 + i * 15}, 70%, 60%)`),
                            }
                        };
                        
                        Plotly.newPlot('category-chart', [trace], {
                            title: 'Top 10 Categories by Revenue',
                            xaxis: { title: 'Category', tickangle: -45 },
                            yaxis: { title: 'Revenue (RMB)' },
                            showlegend: false,
                            margin: {t: 40, b: 120, l: 80, r: 30}
                        });
                    }
                })
                .catch(error => {
                    console.error('Error loading dashboard data:', error);
                    document.getElementById('total-revenue').textContent = 'Error Loading';
                    document.getElementById('total-records').textContent = 'Error Loading';
                    document.getElementById('unique-categories').textContent = 'Error Loading';
                    document.getElementById('avg-revenue').textContent = 'Error Loading';
                });
        </script>
    </body>
    </html>
    """
    
    html_file = os.path.join(data_path, 'dashboard.html')
    with open(html_file, 'w') as f:
        f.write(html_template)
    
    print(f"SUCCESS HTML dashboard generated: {html_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--html':
        generate_html_dashboard()
    else:
        # Run Streamlit dashboard
        dashboard = DashboardViewer()
        dashboard.run_dashboard()