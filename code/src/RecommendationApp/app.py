import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
def main():
    # Set page config
    st.set_page_config(
        page_title="Customer Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )

    # API base URL
    API_BASE_URL = "https://vaibhav84-recommendationapi.hf.space"  # Replace with your API URL

    # Function to make API calls
    def call_api(endpoint, customer_id=None):
        try:
            if customer_id:
                url = f"{API_BASE_URL}/{endpoint}/{customer_id}"
            else:
                url = f"{API_BASE_URL}/{endpoint}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling API: {str(e)}")
            return None

    def get_content_creation(category: str):
        """
        Get AI-generated content for a specific category.
        
        Args:
            category (str): The category for content creation (e.g., 'Investment', 'Trading', 'Banking')
            
        Returns:
            str: Generated content for the specified category
            
        Raises:
            requests.exceptions.RequestException: If the API call fails
            ValueError: If the category is invalid or empty
        """
        try:
            # Input validation
            if not category or not isinstance(category, str):
                raise ValueError("Category must be a non-empty string")

            # Make the API call
            url = f"{API_BASE_URL}/contentcreation/{category}"
            response = requests.get(url)
            # Check if the request was successful
            response.raise_for_status()
            
            # Return the generated content
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error calling content creation API: {str(e)}")
            return None
        except ValueError as e:
            print(f"Invalid input: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None
    

    def hide_sidebar():
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"][aria-expanded="true"]{
                    display: none;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    # Function to login
    def login(customer_id, password):
        try:
            url = f"{API_BASE_URL}/login"
            response = requests.post(url, params={"customer_id": customer_id, "password": password})
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error during login: {str(e)}")
            return None

    # Function to create sentiment distribution chart
    def create_sentiment_chart(sentiment_data):
        fig = go.Figure(data=[
            go.Bar(
                x=list(sentiment_data.keys()),
                y=list(sentiment_data.values()),
                marker_color=['#2ecc71', '#f1c40f', '#e74c3c']
            )
        ])
        fig.update_layout(
            title="Sentiment Distribution",
            xaxis_title="Sentiment Category",
            yaxis_title="Count",
            showlegend=False
        )
        return fig

    # Function to create platform analysis chart
    def create_platform_chart(platform_data):
        df = pd.DataFrame(platform_data)
        fig = px.bar(df, x='platform', y=['post_count', 'avg_sentiment'],
                    barmode='group',
                    title='Platform Analysis')
        return fig
    
        # Add this new function for financial product charts
    def create_investment_projection_chart(initial_amount, monthly_contribution, years, rate):
        months = years * 12
        monthly_rate = rate / 12 / 100
        
        balance = [initial_amount]
        contributions = [initial_amount]
        interest = [0]
        
        for i in range(1, months + 1):
            prev_balance = balance[-1]
            new_contribution = monthly_contribution
            new_interest = (prev_balance + new_contribution) * monthly_rate
            new_balance = prev_balance + new_contribution + new_interest
            
            balance.append(new_balance)
            contributions.append(contributions[-1] + new_contribution)
            interest.append(interest[-1] + new_interest)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(months + 1)),
            y=balance,
            name='Total Balance',
            line=dict(color='#2ECC71', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(months + 1)),
            y=contributions,
            name='Total Contributions',
            line=dict(color='#3498DB', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(months + 1)),
            y=interest,
            name='Total Interest',
            line=dict(color='#E74C3C', width=3)
        ))
        
        fig.update_layout(
            title='Investment Growth Projection',
            xaxis_title='Months',
            yaxis_title='Amount ($)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig

    def create_risk_reward_chart(products):
        fig = px.scatter(
            products,
            x='risk_score',
            y='potential_return',
            size='investment_amount',
            color='category',
            hover_name='name',
            text='name',
            title='Risk vs. Reward Analysis'
        )
        
        fig.update_traces(
            textposition='top center',
            marker=dict(sizeref=2.*max(products['investment_amount'])/100**2)
        )
        
        fig.update_layout(
            xaxis_title='Risk Score',
            yaxis_title='Potential Return (%)',
            template='plotly_white'
        )
        
        return fig
    # Sidebar for login
    st.sidebar.title("Login")
    customer_id = st.sidebar.text_input("Customer ID")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if login_button:
        login_result = login(customer_id, password)
        if login_result and login_result.get("status") == "success":
            st.session_state['logged_in'] = True
            st.session_state['customer_id'] = customer_id
            st.session_state['customer_stats'] = login_result.get("customer_stats", {})
            st.success("Login successful!")
            hide_sidebar()
        else:
            st.error("Login failed. Please check your credentials.")

    # Main content
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        st.title(f"Welcome, Customer {st.session_state['customer_id']}")
        
            # Create tabs for different sections
        # In the main section where tabs are created
        tabs = st.tabs([
            "📊 Dashboard",
            "💭 Social Sentiment",
            "🎯 Recommendations",
            "📈 Predictive Customer Insights",
            "💰 Financial Products",
            "🎨 Multi-Modal Personalization"  # New tab
        ])

        
        # Overview Tab
        # Overview Tab (simplified)
        with tabs[0]:
            st.header("Dashboard")
            
            # Key Metrics
            col1, col2, col3 = st.columns(3)
            stats = st.session_state.get('customer_stats', {})
            
            with col1:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Total Purchases</div>
                        <div class="metric-value">{}</div>
                    </div>
                """.format(stats.get('total_purchases', 0)), unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Total Spent</div>
                        <div class="metric-value">${:,.2f}</div>
                    </div>
                """.format(stats.get('total_spent', 0)), unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Last Purchase</div>
                        <div class="metric-value">{}</div>
                    </div>
                """.format(stats.get('last_purchase_date', 'N/A')), unsafe_allow_html=True)
        
        # Social Sentiment Tab
        with tabs[1]:
            st.header("Sentiment Driven Content Recommendation")
            
            sentiment_data = call_api("social-sentiment", st.session_state['customer_id'])
            if sentiment_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Overall Sentiment")
                    overall = sentiment_data['overall_sentiment']
                    st.metric("Average Score", f"{overall['average_score']:.2f}")
                    st.metric("Recent Score", f"{overall['recent_score']:.2f}")
                    st.metric("Mood", overall['overall_mood'])
                
                with col2:
                    st.subheader("Insights")
                    for insight in sentiment_data['insights']:
                        st.info(insight)
                
                # Sentiment Distribution Chart
                st.plotly_chart(create_sentiment_chart(sentiment_data['sentiment_distribution']))
                
                # Platform Analysis Chart
                st.plotly_chart(create_platform_chart(sentiment_data['platform_analysis']))
                
                # Recent Activities
                st.subheader("Recent Activities")
                for activity in sentiment_data['recent_activities']:
                    with st.expander(f"{activity['platform']} - {activity['timestamp']}"):
                        st.write(f"Content: {activity['content']}")
                        st.write(f"Sentiment Score: {activity['sentiment_score']:.2f}")
                        st.write(f"Intent: {activity['intent']}")
        
        # Recommendations Tab
        # Recommendations Tab
        with tabs[2]:
            st.header("Adaptive Recommendation")
            
            # Create two columns for different types of recommendations
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.subheader("Content Recommendations")
                content_recs = call_api("content-recommendations", st.session_state['customer_id'])
                if content_recs:
                    for rec in content_recs.get('recommendations', []):
                        with st.expander(f"📑 {rec.get('title', 'No Title')} ({rec.get('type', 'Unknown')})"):
                            st.write(f"**Category:** {rec.get('category', 'Unknown')}")
                            content = get_content_creation(rec.get('title', 'No Title'))
                            st.write(content)
                            if 'description' in rec:
                                st.write(f"**Description:** {rec['description']}")
                                
                            if 'url' in rec:
                                st.markdown(f"[View Content]({rec['url']})")
                else:
                    st.info("No content recommendations available at this time.")
            
            with rec_col2:
                st.subheader("Financial Recommendations")
                financial_recs = call_api("financial-recommendations", st.session_state['customer_id'])
                if financial_recs:
                    # Display financial metrics
                    st.metric("Risk Level", financial_recs.get('risk_profile', {}).get('risk_tolerance', 'N/A'))
                    st.metric("Spending Ratio", f"{financial_recs.get('financial_summary', {}).get('spending_ratio', 0):.1f}%")
                    
                    # Display action items
                    st.markdown("### Action Items")
                    for item in financial_recs.get('recommendations', {}).get('action_items', []):
                        with st.expander(f"🎯 {item['action']} (Priority: {item['priority']})"):
                            st.write(f"**Impact:** {item['impact']}")
                            st.write(f"**Timeline:** {item['timeline']}")
                else:
                    st.info("No financial recommendations available at this time.")


        # Add this new tab section after your existing tabs
        with tabs[4]:  # Financial Products Tab
            st.header("Hyper-Personalized Financial Products")
            
            # Get financial data
            financial_data = call_api("financial-recommendations", st.session_state['customer_id'])
            
            if financial_data:
                # Customer Financial Profile Section
                st.subheader("Your Financial Profile")
                profile_cols = st.columns(4)
                
                with profile_cols[0]:
                    risk_tolerance = financial_data.get('risk_profile', {}).get('risk_tolerance', 'Moderate')
                    st.metric("Risk Tolerance", risk_tolerance)
                
                with profile_cols[1]:
                    investment_horizon = financial_data.get('risk_profile', {}).get('investment_horizon', '5-10 years')
                    st.metric("Investment Horizon", investment_horizon)
                
                with profile_cols[2]:
                    monthly_savings = financial_data.get('financial_summary', {}).get('monthly_savings', 0)
                    st.metric("Monthly Savings Potential", f"${monthly_savings:,.2f}")
                
                with profile_cols[3]:
                    investment_capacity = financial_data.get('financial_summary', {}).get('investment_capacity', 0)
                    st.metric("Investment Capacity", f"${investment_capacity:,.2f}")
                
                # Investment Simulator Section
                st.subheader("Investment Growth Simulator")
                sim_cols = st.columns([2, 1])
                
                with sim_cols[1]:
                    st.markdown("### Adjust Parameters")
                    initial_investment = st.number_input(
                        "Initial Investment ($)",
                        min_value=0,
                        max_value=1000000,
                        value=10000,
                        step=1000
                    )
                    
                    monthly_contribution = st.number_input(
                        "Monthly Contribution ($)",
                        min_value=0,
                        max_value=10000,
                        value=500,
                        step=100
                    )
                    
                    investment_years = st.slider(
                        "Investment Period (Years)",
                        min_value=1,
                        max_value=30,
                        value=10
                    )
                    
                    expected_return = st.slider(
                        "Expected Annual Return (%)",
                        min_value=1,
                        max_value=15,
                        value=7
                    )
                
                with sim_cols[0]:
                    projection_chart = create_investment_projection_chart(
                        initial_investment,
                        monthly_contribution,
                        investment_years,
                        expected_return
                    )
                    st.plotly_chart(projection_chart, use_container_width=True)
                
                # Personalized Product Recommendations
                st.subheader("Recommended Financial Products")
        #---------------------------------
        # Add this after your existing tabs code
        with tabs[5]:  # Multi-modal Personalization Tab
            st.header("Multi-modal Personalization Hub")
            
            # Get existing data from APIs
            sentiment_data = call_api("social-sentiment", st.session_state['customer_id'])
            financial_data = call_api("financial-recommendations", st.session_state['customer_id'])
            
            if sentiment_data and financial_data:
                # Interactive Profile Section
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Engagement Score Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=sentiment_data['overall_sentiment']['average_score'] * 100,
                        title={'text': "Engagement Score"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [0, 100]},
                            'bar': {'color': "#2ECC71"},
                            'steps': [
                                {'range': [0, 33], 'color': "#FF5733"},
                                {'range': [33, 66], 'color': "#FFC300"},
                                {'range': [66, 100], 'color': "#2ECC71"}
                            ]}
                    ))
                    fig_gauge.update_layout(height=250)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                with col2:
                    # Key Metrics with Trend Indicators
                    current_mood = sentiment_data['overall_sentiment']['overall_mood']
                    mood_color = "#2ECC71" if current_mood == "Positive" else "#E74C3C"
                    
                    st.markdown(f"""
                        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
                            <h3 style='color: {mood_color};'>{current_mood} Mood</h3>
                            <p>Based on recent interactions</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Interactive Preference Center
                st.subheader("Personalization Preferences")
                pref_cols = st.columns(3)
                
                with pref_cols[0]:
                    communication_pref = st.selectbox(
                        "Communication Channel",
                        ["Email", "SMS", "Push Notifications", "All Channels"],
                        help="Choose your preferred communication channel"
                    )
                
                with pref_cols[1]:
                    update_frequency = st.select_slider(
                        "Update Frequency",
                        options=["Daily", "Weekly", "Bi-weekly", "Monthly"],
                        value="Weekly"
                    )
                
                with pref_cols[2]:
                    notification_types = st.multiselect(
                        "Notification Types",
                        ["Product Updates", "Financial Tips", "Market Insights", "Account Alerts"],
                        ["Account Alerts"]
                    )
                
                # Interactive Timeline Alternative
                st.subheader("Engagement Timeline")
                activities = sentiment_data.get('recent_activities', [])
                if activities:
                    # Create a more compact visualization
                    df_activities = pd.DataFrame(activities)
                    df_activities['timestamp'] = pd.to_datetime(df_activities['timestamp'])
                    
                    # Sort by timestamp
                    df_activities = df_activities.sort_values('timestamp')
                    
                    # Create an interactive scatter plot timeline
                    fig = px.scatter(
                        df_activities,
                        x='timestamp',
                        y='platform',
                        color='sentiment_score',
                        size=[10] * len(df_activities),  # Constant size
                        hover_data=['content'],
                        color_continuous_scale='RdYlGn',  # Red to Yellow to Green scale
                        title='User Engagement Timeline'
                    )
                    
                    # Customize the appearance
                    fig.update_layout(
                        height=300,
                        xaxis_title="Time",
                        yaxis_title="Platform",
                        showlegend=False,
                        template="plotly_white"
                    )
                    
                    # Add hover template
                    fig.update_traces(
                        hovertemplate="<b>%{y}</b><br>" +
                                    "Time: %{x}<br>" +
                                    "Content: %{customdata[0]}<br>" +
                                    "Sentiment: %{marker.color:.2f}<extra></extra>"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interactive filters
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_platform = st.multiselect(
                            "Filter by Platform",
                            options=df_activities['platform'].unique(),
                            default=df_activities['platform'].unique()
                        )
                    
                    with col2:
                        date_range = st.date_input(
                            "Select Date Range",
                            value=(df_activities['timestamp'].min().date(),
                                df_activities['timestamp'].max().date())
                        )
                    
                    # Display filtered activities
                    filtered_df = df_activities[
                        (df_activities['platform'].isin(selected_platform)) &
                        (df_activities['timestamp'].dt.date.between(date_range[0], date_range[1]))
                    ]
                    
                    # Display activities in an expander
                    with st.expander("View Detailed Activities"):
                        for _, activity in filtered_df.iterrows():
                            st.markdown(f"""
                                <div style='
                                    padding: 10px;
                                    border-left: 3px solid {
                                        '#2ECC71' if activity['sentiment_score'] > 0.5
                                        else '#E74C3C' if activity['sentiment_score'] < -0.5
                                        else '#F1C40F'
                                    };
                                    margin: 5px 0;
                                    background-color: white;
                                    border-radius: 5px;
                                '>
                                    <small>{activity['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
                                    <br>
                                    <strong>{activity['platform']}</strong>
                                    <p>{activity['content']}</p>
                                </div>
                            """, unsafe_allow_html=True)

                
                # Personalized Insights Cards
                st.subheader("Personalized Insights")
                insight_cols = st.columns(3)
                
                insights = [
                    {
                        "title": "Financial Behavior",
                        "value": financial_data.get('risk_profile', {}).get('risk_tolerance', 'Moderate'),
                        "icon": "💰",
                        "color": "#3498db"
                    },
                    {
                        "title": "Social Engagement",
                        "value": f"{len(activities)} interactions",
                        "icon": "🤝",
                        "color": "#2ecc71"
                    },
                    {
                        "title": "Sentiment Trend",
                        "value": sentiment_data['overall_sentiment']['recent_score'],
                        "icon": "📈",
                        "color": "#e74c3c"
                    }
                ]
                
                for col, insight in zip(insight_cols, insights):
                    with col:
                        st.markdown(f"""
                            <div style='
                                background-color: white;
                                padding: 20px;
                                border-radius: 10px;
                                border-left: 5px solid {insight["color"]};
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            '>
                                <h1 style='font-size: 2em; margin: 0;'>{insight["icon"]}</h1>
                                <h3 style='color: {insight["color"]};'>{insight["title"]}</h3>
                                <p style='font-size: 1.2em; font-weight: bold;'>{insight["value"]}</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Interactive Recommendation Section
                st.subheader("Smart Recommendations")
                rec_tabs = st.tabs(["Products", "Services", "Educational Content"])
                
                with rec_tabs[0]:
                    product_cols = st.columns(2)
                    for i, product in enumerate(financial_data.get('recommendations', {}).get('products', [])):
                        with product_cols[i % 2]:
                            st.markdown(f"""
                                <div style='
                                    background-color: white;
                                    padding: 20px;
                                    border-radius: 10px;
                                    margin: 10px 0;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                '>
                                    <h4>{product.get('name', 'Product')}</h4>
                                    <p>{product.get('description', '')}</p>
                                    <div class='recommendation-score'>
                                        Match Score: {product.get('match_score', 0)}%
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)


        #---------------------------------        
        # Full Analysis Tab
        # Full Analysis Tab
        with tabs[3]:
            st.header("Comprehensive Analysis")
        
        full_analysis = call_api("customer-analysis", st.session_state['customer_id'])
        if full_analysis:
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Customer Profile Summary
                st.subheader("Customer Profile")
                profile_metrics = st.columns(3)
                
                with profile_metrics[0]:
                    st.metric(
                        "Sentiment Score",
                        f"{full_analysis.get('sentiment_analysis', {}).get('overall_sentiment', {}).get('average_score', 0):.2f}",
                        f"{full_analysis.get('sentiment_analysis', {}).get('overall_sentiment', {}).get('recent_score', 0):.2f} recent"
                    )
                
                with profile_metrics[1]:
                    st.metric(
                        "Total Posts",
                        sum(p.get('post_count', 0) for p in full_analysis.get('sentiment_analysis', {}).get('platform_analysis', []))
                    )
                
                with profile_metrics[2]:
                    st.metric(
                        "Overall Mood",
                        full_analysis.get('sentiment_analysis', {}).get('overall_sentiment', {}).get('overall_mood', 'N/A')
                    )

                # Sentiment Trend Analysis
                st.subheader("Sentiment Analysis Over Time")
                recent_activities = full_analysis.get('sentiment_analysis', {}).get('recent_activities', [])
                if recent_activities:
                    df_activities = pd.DataFrame(recent_activities)
                    df_activities['timestamp'] = pd.to_datetime(df_activities['timestamp'])
                    
                    fig_trend = px.line(
                        df_activities,
                        x='timestamp',
                        y='sentiment_score',
                        title='Sentiment Trend',
                        line_shape='spline'
                    )
                    fig_trend.update_layout(height=400)
                    st.plotly_chart(fig_trend, use_container_width=True)

                # Platform Comparison
                st.subheader("Platform Analysis")
                platform_data = full_analysis.get('sentiment_analysis', {}).get('platform_analysis', [])
                if platform_data:
                    df_platform = pd.DataFrame(platform_data)
                    
                    fig_platform = px.bar(
                        df_platform,
                        x='platform',
                        y=['post_count', 'avg_sentiment'],
                        title='Platform Comparison',
                        barmode='group'
                    )
                    fig_platform.update_layout(height=400)
                    st.plotly_chart(fig_platform, use_container_width=True)

            with col2:
                # Key Insights
                st.subheader("Key Insights")
                insights = full_analysis.get('sentiment_analysis', {}).get('insights', [])
                for insight in insights:
                    st.info(insight)

                # Intent Distribution
                st.subheader("Intent Distribution")
                intent_data = full_analysis.get('sentiment_analysis', {}).get('intent_analysis', {})
                if intent_data:
                    fig_intent = px.pie(
                        values=list(intent_data.values()),
                        names=list(intent_data.keys()),
                        title='Customer Intent Distribution'
                    )
                    fig_intent.update_layout(height=300)
                    st.plotly_chart(fig_intent, use_container_width=True)

            # Recommendations Summary
            st.subheader("Recommendations Summary")
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                # Content Recommendations
                st.markdown("### Content Recommendations")
                content_recs = full_analysis.get('content_recommendations', {}).get('recommendations', [])
                for rec in content_recs:
                    with st.expander(f"📚 {rec.get('title', 'No Title')}"):
                        st.write(f"**Type:** {rec.get('type', 'N/A')}")
                        st.write(f"**Category:** {rec.get('category', 'N/A')}")
                        if 'description' in rec:
                            st.write(f"**Description:** {rec['description']}")
                        if 'url' in rec:
                            st.markdown(f"[View Content]({rec['url']})")

            with rec_col2:
                # Purchase Recommendations
                st.markdown("### Purchase Recommendations")
                purchase_recs = full_analysis.get('purchase_recommendations', {}).get('recommendations', [])
                for rec in purchase_recs:
                    with st.expander(f"🛍️ Product {rec.get('product_id', 'N/A')}"):
                        st.write(f"**Category:** {rec.get('category', 'N/A')}")
                        if 'confidence' in rec:
                            st.progress(float(rec['confidence']))
                            st.write(f"Confidence: {rec['confidence']:.2%}")

            # Recent Activity Timeline
            st.subheader("Recent Activity Timeline")
            activities = full_analysis.get('sentiment_analysis', {}).get('recent_activities', [])
            if activities:
                for activity in activities:
                    with st.expander(
                        f"🕒 {activity.get('timestamp', 'N/A')} | {activity.get('platform', 'N/A')}"
                    ):
                        st.write(f"**Content:** {activity.get('content', 'N/A')}")
                        sentiment_score = activity.get('sentiment_score', 0)
                        sentiment_color = (
                            'green' if sentiment_score > 0.5
                            else 'red' if sentiment_score < -0.5
                            else 'orange'
                        )
                        st.markdown(
                            f"**Sentiment Score:** <span style='color:{sentiment_color}'>"
                            f"{sentiment_score:.2f}</span>",
                            unsafe_allow_html=True
                        )
                        st.write(f"**Intent:** {activity.get('intent', 'N/A')}")

            # Add download button for full analysis
            st.download_button(
                "Download Full Analysis",
                json.dumps(full_analysis, indent=2),
                "customer_analysis.json",
                "application/json"
            )

        else:
            st.error("Unable to load customer analysis data")

         # Sample product data (replace with API data in production)
        products_data = {
            'name': ['Conservative Fund', 'Balanced Fund', 'Growth Fund', 'High-Yield Savings', 'Index Fund'],
            'category': ['Mutual Fund', 'Mutual Fund', 'Mutual Fund', 'Savings', 'ETF'],
            'risk_score': [2, 5, 8, 1, 6],
            'potential_return': [5, 8, 12, 3, 9],
            'investment_amount': [10000, 15000, 20000, 5000, 12000]
        }
        products_df = pd.DataFrame(products_data)
        
        # Risk-Reward Analysis
        st.plotly_chart(create_risk_reward_chart(products_df), use_container_width=True)
        
        # Product Details
        product_cols = st.columns(3)
        for i, product in products_df.iterrows():
            with product_cols[i % 3]:
                with st.expander(f"📈 {product['name']}"):
                    st.write(f"**Category:** {product['category']}")
                    st.write(f"**Risk Score:** {product['risk_score']}/10")
                    st.write(f"**Potential Return:** {product['potential_return']}%")
                    st.write(f"**Recommended Investment:** ${product['investment_amount']:,}")
                    st.progress(product['risk_score']/10)
                    if st.button(f"Learn More about {product['name']}", key=f"learn_more_{i}"):
                        st.write("Detailed information would appear here...")
        
        # Portfolio Optimization Section
        st.subheader("Portfolio Optimization")
        opt_cols = st.columns(2)
        
        with opt_cols[0]:
            st.markdown("### Current Asset Allocation")
            current_allocation = {
                'Stocks': 40,
                'Bonds': 30,
                'Cash': 20,
                'Real Estate': 10
            }
            
            fig_current = go.Figure(data=[go.Pie(
                labels=list(current_allocation.keys()),
                values=list(current_allocation.values()),
                hole=.3
            )])
            fig_current.update_layout(title='Current Portfolio')
            st.plotly_chart(fig_current, use_container_width=True)
        
        with opt_cols[1]:
            st.markdown("### Recommended Asset Allocation")
            recommended_allocation = {
                'Stocks': 50,
                'Bonds': 25,
                'Cash': 15,
                'Real Estate': 10
            }
            
            fig_recommended = go.Figure(data=[go.Pie(
                labels=list(recommended_allocation.keys()),
                values=list(recommended_allocation.values()),
                hole=.3
            )])
            fig_recommended.update_layout(title='Recommended Portfolio')
            st.plotly_chart(fig_recommended, use_container_width=True)

        
        # Logout button
        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

    else:
        st.title("Welcome to Customer Analytics Dashboard")
        st.write("Please login to view your personalized dashboard.")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<p style='color:gray; font-weight:bold;'>Join us in building cutting-edge AI solutions that transform customer experiences by delivering actionable insights, optimizing customer engagement, and driving AI innovation to new heights.</p>",
            unsafe_allow_html=True
        )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Dashboard v1.0")
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .stExpander {
            background-color: #ffffff;
            border: 1px solid #e6e6e6;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .stProgress > div > div > div {
            background-color: #3498db;
        }
        </style>
    """, unsafe_allow_html=True)
            # Add custom CSS for this tab
    st.markdown("""
            <style>
            .financial-product-card {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 10px 0;
            }
            .risk-meter {
                height: 10px;
                background: linear-gradient(to right, #2ECC71, #F1C40F, #E74C3C);
                border-radius: 5px;
                margin: 10px 0;
            }
            </style>
        """, unsafe_allow_html=True)
    st.markdown("""
            <style>
            .recommendation-score {
                color: #2ecc71;
                font-weight: bold;
                margin-top: 10px;
            }
            .stSelectbox {
                background-color: white;
                border-radius: 5px;
            }
            .stMultiSelect {
                background-color: white;
                border-radius: 5px;
            }
            </style>
        """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()