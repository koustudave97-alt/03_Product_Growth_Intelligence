import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PRODUCT & GROWTH INTELLIGENCE PLATFORM
# FINAL OPTIMIZED STREAMLIT APPLICATION
# ============================================================

st.set_page_config(
    page_title="Product & Growth Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APPLICATION STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #6B7280;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            padding: 14px;
            border-radius: 12px;
        }

        div[data-testid="stSidebar"] {
            background-color: #F8FAFC;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# DATA PREPARATION
# ============================================================

def ensure_event_flags(data):

    data = data.copy()

    if "event" in data.columns:

        event_name = data["event"].astype(str).str.lower()

        if "is_view" not in data.columns:
            data["is_view"] = (
                event_name
                .eq("view")
                .astype("int8")
            )

        if "is_addtocart" not in data.columns:
            data["is_addtocart"] = (
                event_name
                .eq("addtocart")
                .astype("int8")
            )

        if "is_transaction" not in data.columns:
            data["is_transaction"] = (
                event_name
                .eq("transaction")
                .astype("int8")
            )

    return data


# ============================================================
# FAST DATA LOADING
# ============================================================

@st.cache_resource
def load_data():

    # Load only datasets actually used by this application.
    # This improves startup performance.

    events = pd.read_csv(
        DATA_DIR / "events_clean.csv",
        low_memory=False
    )

    events_enriched = pd.read_csv(
        DATA_DIR / "events_enriched.csv",
        low_memory=False
    )

    events = ensure_event_flags(events)
    events_enriched = ensure_event_flags(events_enriched)

    for data in [events, events_enriched]:

        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce"
        )

        data.dropna(
            subset=["visitorid", "date"],
            inplace=True
        )

        # Precompute normalized date once.
        data["date_day"] = (
            data["date"]
            .dt
            .normalize()
        )

    return events, events_enriched


try:
    events, events_enriched = (
        load_data()
    )

except Exception as error:

    st.error(
        "Unable to load the processed project datasets."
    )

    st.exception(error)

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_divide(numerator, denominator):

    if denominator is None or denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def percentage(value):

    return f"{float(value) * 100:.2f}%"


def integer_format(value):

    return f"{int(round(float(value))):,}"


def get_previous_period(
    start_date,
    end_date,
    minimum_date
):

    period_days = (
        end_date - start_date
    ).days + 1

    previous_end = (
        start_date
        - pd.Timedelta(days=1)
    )

    previous_start = (
        previous_end
        - pd.Timedelta(
            days=period_days - 1
        )
    )

    if previous_start < minimum_date:
        previous_start = minimum_date

    return previous_start, previous_end


def get_category_column(data):

    possible_columns = [
        "categoryid",
        "category_id",
        "category",
        "category_name",
        "property_value"
    ]

    for column in possible_columns:

        if column in data.columns:
            return column

    return None


def calculate_period_metrics(data):

    if data.empty:

        return {
            "events": 0,
            "visitors": 0,
            "views": 0,
            "carts": 0,
            "transactions": 0,
            "view_to_cart": 0.0,
            "cart_to_transaction": 0.0,
            "overall_conversion": 0.0
        }

    total_events = len(data)

    total_visitors = (
        data["visitorid"]
        .nunique()
    )

    total_views = int(
        data["is_view"]
        .sum()
    )

    total_carts = int(
        data["is_addtocart"]
        .sum()
    )

    total_transactions = int(
        data["is_transaction"]
        .sum()
    )

    return {
        "events": total_events,
        "visitors": total_visitors,
        "views": total_views,
        "carts": total_carts,
        "transactions": total_transactions,
        "view_to_cart": safe_divide(
            total_carts,
            total_views
        ),
        "cart_to_transaction": safe_divide(
            total_transactions,
            total_carts
        ),
        "overall_conversion": safe_divide(
            total_transactions,
            total_views
        )
    }


def filter_events(
    data,
    start_date,
    end_date,
    selected_visitors=None
):

    mask = (
        data["date_day"]
        .between(
            start_date,
            end_date
        )
    )

    if selected_visitors is not None:

        mask &= (
            data["visitorid"]
            .isin(selected_visitors)
        )

    return data.loc[mask]


def add_conversion_rates(data):

    data = data.copy()

    data["view_to_cart_rate"] = np.divide(
        data["add_to_carts"].to_numpy(
            dtype=float
        ),
        data["views"].to_numpy(
            dtype=float
        ),
        out=np.zeros(
            len(data),
            dtype=float
        ),
        where=(
            data["views"]
            .to_numpy(dtype=float)
            != 0
        )
    )

    data["cart_to_transaction_rate"] = np.divide(
        data["transactions"].to_numpy(
            dtype=float
        ),
        data["add_to_carts"].to_numpy(
            dtype=float
        ),
        out=np.zeros(
            len(data),
            dtype=float
        ),
        where=(
            data["add_to_carts"]
            .to_numpy(dtype=float)
            != 0
        )
    )

    data["overall_conversion_rate"] = np.divide(
        data["transactions"].to_numpy(
            dtype=float
        ),
        data["views"].to_numpy(
            dtype=float
        ),
        out=np.zeros(
            len(data),
            dtype=float
        ),
        where=(
            data["views"]
            .to_numpy(dtype=float)
            != 0
        )
    )

    return data


# ============================================================
# FAST VISITOR FIRST ACTIVITY
# ============================================================

@st.cache_data(show_spinner=False)
def get_visitor_first_activity(data):

    return (
        data
        .groupby(
            "visitorid",
            sort=False
        )["date_day"]
        .min()
        .rename(
            "first_activity_date"
        )
        .reset_index()
    )


# ============================================================
# FAST BEHAVIORAL SEGMENTATION
# BUILT ONLY WHEN NEEDED
# ============================================================

@st.cache_data(show_spinner=False)
def build_visitor_segments(full_events):

    required_columns = [
        "visitorid",
        "date_day",
        "event",
        "is_view",
        "is_addtocart",
        "is_transaction"
    ]

    if "transactionid" in full_events.columns:
        required_columns.append(
            "transactionid"
        )

    data = (
        full_events[
            required_columns
        ]
        .copy()
    )

    analysis_end_date = (
        data["date_day"]
        .max()
    )

    # Main visitor features

    visitor_features = (
        data
        .groupby(
            "visitorid",
            sort=False
        )
        .agg(
            first_activity_date=(
                "date_day",
                "min"
            ),
            last_activity_date=(
                "date_day",
                "max"
            ),
            total_events=(
                "event",
                "size"
            ),
            view_events=(
                "is_view",
                "sum"
            ),
            cart_events=(
                "is_addtocart",
                "sum"
            ),
            transaction_events=(
                "is_transaction",
                "sum"
            )
        )
        .reset_index()
    )

    # Active days without slow lambda functions

    active_days = (
        data
        .groupby(
            "visitorid",
            sort=False
        )["date_day"]
        .nunique()
        .rename(
            "active_days"
        )
        .reset_index()
    )

    visitor_features = (
        visitor_features
        .merge(
            active_days,
            on="visitorid",
            how="left",
            sort=False
        )
    )

    # Unique transactions

    if "transactionid" in data.columns:

        transaction_data = (
            data.loc[
                (
                    data["is_transaction"]
                    == 1
                )
                &
                (
                    data["transactionid"]
                    .notna()
                ),
                [
                    "visitorid",
                    "transactionid"
                ]
            ]
        )

        if not transaction_data.empty:

            unique_transactions = (
                transaction_data
                .groupby(
                    "visitorid",
                    sort=False
                )["transactionid"]
                .nunique()
                .rename(
                    "unique_transactions"
                )
                .reset_index()
            )

            visitor_features = (
                visitor_features
                .merge(
                    unique_transactions,
                    on="visitorid",
                    how="left",
                    sort=False
                )
            )

        else:

            visitor_features[
                "unique_transactions"
            ] = (
                visitor_features[
                    "transaction_events"
                ]
            )

    else:

        visitor_features[
            "unique_transactions"
        ] = (
            visitor_features[
                "transaction_events"
            ]
        )

    visitor_features[
        "unique_transactions"
    ] = (
        visitor_features[
            "unique_transactions"
        ]
        .fillna(0)
        .astype(int)
    )

    # Recency

    visitor_features[
        "recency_days"
    ] = (
        analysis_end_date
        - visitor_features[
            "last_activity_date"
        ]
    ).dt.days

    # Data-driven thresholds

    active_day_threshold = max(
        2,
        int(
            np.ceil(
                visitor_features[
                    "active_days"
                ]
                .quantile(0.75)
            )
        )
    )

    view_event_threshold = max(
        5,
        int(
            np.ceil(
                visitor_features[
                    "view_events"
                ]
                .quantile(0.75)
            )
        )
    )

    cart_users = (
        visitor_features.loc[
            visitor_features[
                "cart_events"
            ] > 0,
            "cart_events"
        ]
    )

    if not cart_users.empty:

        high_intent_cart_threshold = max(
            2,
            int(
                np.ceil(
                    cart_users.quantile(0.75)
                )
            )
        )

    else:

        high_intent_cart_threshold = 2

    new_visitor_cutoff = (
        analysis_end_date
        - pd.Timedelta(days=6)
    )

    # Mutually exclusive segments.
    # np.select uses first matching condition.

    visitor_features["segment"] = np.select(
        [

            visitor_features[
                "unique_transactions"
            ] >= 2,

            visitor_features[
                "unique_transactions"
            ] == 1,

            (
                (
                    visitor_features[
                        "transaction_events"
                    ] == 0
                )
                &
                (
                    visitor_features[
                        "cart_events"
                    ]
                    >= high_intent_cart_threshold
                )
            ),

            (
                (
                    visitor_features[
                        "transaction_events"
                    ] == 0
                )
                &
                (
                    visitor_features[
                        "cart_events"
                    ] > 0
                )
                &
                (
                    visitor_features[
                        "cart_events"
                    ]
                    < high_intent_cart_threshold
                )
            ),

            (
                (
                    visitor_features[
                        "transaction_events"
                    ] == 0
                )
                &
                (
                    visitor_features[
                        "active_days"
                    ] > 1
                )
                &
                (
                    visitor_features[
                        "recency_days"
                    ] >= 14
                )
            ),

            (
                visitor_features[
                    "first_activity_date"
                ]
                >= new_visitor_cutoff
            ),

            (
                (
                    visitor_features[
                        "cart_events"
                    ] == 0
                )
                &
                (
                    visitor_features[
                        "transaction_events"
                    ] == 0
                )
                &
                (
                    visitor_features[
                        "active_days"
                    ]
                    >= active_day_threshold
                )
                &
                (
                    visitor_features[
                        "view_events"
                    ]
                    >= view_event_threshold
                )
            )

        ],

        [

            "Repeat Buyers",

            "One-Time Buyers",

            "High-Intent Visitors",

            "Cart Abandoners",

            "At-Risk Visitors",

            "New Visitors",

            "Highly Engaged Browsers"

        ],

        default="Other Visitors"
    )

    thresholds = {
        "active_day_threshold": (
            active_day_threshold
        ),
        "view_event_threshold": (
            view_event_threshold
        ),
        "high_intent_cart_threshold": (
            high_intent_cart_threshold
        ),
        "new_visitor_cutoff": (
            new_visitor_cutoff
        ),
        "analysis_end_date": (
            analysis_end_date
        )
    }

    return visitor_features, thresholds


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📈 Growth Intelligence"
)

st.sidebar.caption(
    "Product & Growth Intelligence Platform"
)

st.sidebar.divider()


# ============================================================
# ANALYSIS CONTROLS
# ============================================================

st.sidebar.subheader(
    "🔍 Analysis Controls"
)

min_date = (
    events["date_day"]
    .min()
)

max_date = (
    events["date_day"]
    .max()
)


period_option = st.sidebar.selectbox(
    "Quick Period",
    [
        "Full Dataset",
        "Last 7 Days",
        "Last 14 Days",
        "Last 30 Days",
        "Last 60 Days",
        "Custom Range"
    ]
)


def get_quick_period(option):

    if option == "Full Dataset":

        return (
            min_date,
            max_date
        )

    days_map = {
        "Last 7 Days": 7,
        "Last 14 Days": 14,
        "Last 30 Days": 30,
        "Last 60 Days": 60
    }

    days = days_map[option]

    calculated_start = max(
        min_date,
        max_date
        - pd.Timedelta(
            days=days - 1
        )
    )

    return (
        calculated_start,
        max_date
    )


if period_option == "Custom Range":

    selected_dates = st.sidebar.date_input(
        "Select Analysis Period",
        value=(
            min_date.date(),
            max_date.date()
        ),
        min_value=min_date.date(),
        max_value=max_date.date(),
        format="DD/MM/YYYY"
    )

    if (
        isinstance(
            selected_dates,
            tuple
        )
        and len(selected_dates) == 2
    ):

        start_date = (
            pd.Timestamp(
                selected_dates[0]
            )
            .normalize()
        )

        end_date = (
            pd.Timestamp(
                selected_dates[1]
            )
            .normalize()
        )

    else:

        start_date = min_date
        end_date = max_date

else:

    start_date, end_date = (
        get_quick_period(
            period_option
        )
    )

    st.sidebar.caption(
        f"Selected: "
        f"{start_date.date()} → "
        f"{end_date.date()}"
    )


# ============================================================
# USER GROUP FILTER
# ============================================================

segment_options = [

    "All Users",

    "Repeat Buyers",

    "One-Time Buyers",

    "High-Intent Visitors",

    "Cart Abandoners",

    "At-Risk Visitors",

    "New Visitors",

    "Highly Engaged Browsers",

    "Other Visitors"

]


selected_user_group = st.sidebar.selectbox(
    "Select User Group",
    segment_options
)


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [

        "🏠 Growth Overview",

        "🆕 New Visitor Monitor",

        "🔻 Funnel Investigator",

        "🔄 Retention & Cohorts",

        "👥 User Segmentation",

        "📦 Product & Category Intelligence",

        "🧪 Growth Decision Lab",

        "🎯 Growth Opportunities"

    ]
)


# ============================================================
# LAZY SEGMENTATION
# ============================================================

segment_dependent_pages = {

    "👥 User Segmentation",

    "🎯 Growth Opportunities"

}


needs_segments = (

    selected_user_group != "All Users"

    or page in segment_dependent_pages

)


visitor_segments = None
segment_thresholds = None
selected_visitors = None


if needs_segments:

    visitor_segments, segment_thresholds = (
        build_visitor_segments(
            events
        )
    )

    if selected_user_group != "All Users":

        selected_visitors = set(
            visitor_segments.loc[
                visitor_segments[
                    "segment"
                ]
                == selected_user_group,
                "visitorid"
            ]
        )


# ============================================================
# APPLY CORE FILTER
# ============================================================

filtered_events = filter_events(
    events,
    start_date,
    end_date,
    selected_visitors
)


current_metrics = (
    calculate_period_metrics(
        filtered_events
    )
)


total_events = current_metrics["events"]

total_visitors = current_metrics["visitors"]

total_views = current_metrics["views"]

total_add_to_carts = current_metrics["carts"]

total_transactions = (
    current_metrics["transactions"]
)

view_to_cart_rate = (
    current_metrics["view_to_cart"]
)

cart_to_transaction_rate = (
    current_metrics[
        "cart_to_transaction"
    ]
)

overall_conversion_rate = (
    current_metrics[
        "overall_conversion"
    ]
)


# ============================================================
# LAZY NEW VISITOR ANALYSIS
# ============================================================

new_visitor_pages = {

    "🏠 Growth Overview",

    "🆕 New Visitor Monitor",

    "🧪 Growth Decision Lab",

    "🎯 Growth Opportunities"

}


new_visitors_in_period = pd.DataFrame()
new_visitor_events = pd.DataFrame()

new_visitor_count = 0
new_visitor_transactions = 0

new_visitor_conversion = 0.0
new_visitor_share = 0.0


if page in new_visitor_pages:

    visitor_first_activity = (
        get_visitor_first_activity(
            events
        )
    )

    new_visitor_mask = (
        visitor_first_activity[
            "first_activity_date"
        ]
        .between(
            start_date,
            end_date
        )
    )

    if selected_visitors is not None:

        new_visitor_mask &= (
            visitor_first_activity[
                "visitorid"
            ]
            .isin(selected_visitors)
        )

    new_visitors_in_period = (
        visitor_first_activity.loc[
            new_visitor_mask
        ]
    )

    new_visitor_ids = set(
        new_visitors_in_period[
            "visitorid"
        ]
    )

    new_visitor_count = len(
        new_visitors_in_period
    )

    if new_visitor_ids:

        new_visitor_events = (
            filtered_events.loc[
                filtered_events[
                    "visitorid"
                ]
                .isin(
                    new_visitor_ids
                )
            ]
        )

        new_visitor_transactions = int(
            new_visitor_events[
                "is_transaction"
            ]
            .sum()
        )

        new_visitor_conversion = (
            safe_divide(
                new_visitor_transactions,
                int(
                    new_visitor_events[
                        "is_view"
                    ]
                    .sum()
                )
            )
        )

    new_visitor_share = (
        safe_divide(
            new_visitor_count,
            total_visitors
        )
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📈 Product & Growth Intelligence Platform'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive decision-support application for investigating '
    'user behavior, acquisition, funnel performance, retention, '
    'product engagement, and growth opportunities.'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Analysis period: "
    f"{start_date.date()} → "
    f"{end_date.date()} | "
    f"User scope: "
    f"{selected_user_group}"
)

st.divider()


# ============================================================
# PAGE 1 — GROWTH OVERVIEW
# ============================================================

if page == "🏠 Growth Overview":

    st.header(
        "Growth Overview"
    )

    st.write(
        "Monitor the health of the selected user population "
        "and compare current activity with the previous "
        "available period."
    )

    previous_start, previous_end = (
        get_previous_period(
            start_date,
            end_date,
            min_date
        )
    )

    previous_events = (
        filter_events(
            events,
            previous_start,
            previous_end,
            selected_visitors
        )
    )

    previous_metrics = (
        calculate_period_metrics(
            previous_events
        )
    )

    visitor_delta = (
        total_visitors
        - previous_metrics["visitors"]
    )

    event_delta = (
        total_events
        - previous_metrics["events"]
    )

    transaction_delta = (
        total_transactions
        - previous_metrics["transactions"]
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Total Visitors",
        integer_format(
            total_visitors
        ),
        delta=visitor_delta
    )

    col2.metric(
        "Total Events",
        integer_format(
            total_events
        ),
        delta=event_delta
    )

    col3.metric(
        "Add-to-Cart Events",
        integer_format(
            total_add_to_carts
        )
    )

    col4.metric(
        "Transactions",
        integer_format(
            total_transactions
        ),
        delta=transaction_delta
    )

    st.divider()

    st.subheader(
        "Acquisition & New Users"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "New Visitors",
        integer_format(
            new_visitor_count
        )
    )

    col2.metric(
        "New Visitor Share",
        percentage(
            new_visitor_share
        )
    )

    col3.metric(
        "New Visitor Conversion",
        percentage(
            new_visitor_conversion
        )
    )

    st.caption(
        "New Visitors are users whose first observed activity "
        "falls inside the selected analysis period."
    )

    st.divider()

    st.subheader(
        "Conversion Performance"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "View → Cart",
        percentage(
            view_to_cart_rate
        )
    )

    col2.metric(
        "Cart → Transaction",
        percentage(
            cart_to_transaction_rate
        )
    )

    col3.metric(
        "Overall Conversion",
        percentage(
            overall_conversion_rate
        )
    )

    st.divider()

    st.subheader(
        "Activity Trend"
    )

    daily_activity = (
        filtered_events
        .groupby(
            "date_day",
            sort=True
        )
        .size()
        .rename(
            "events"
        )
    )

    if not daily_activity.empty:

        st.line_chart(
            daily_activity
        )

    else:

        st.info(
            "No activity is available for the selected filters."
        )

    st.divider()

    st.subheader(
        "Key Finding"
    )

    if total_views == 0:

        st.info(
            "No product views are available for the selected filters."
        )

    else:

        first_stage_loss = (
            1
            - view_to_cart_rate
        )

        second_stage_loss = (
            1
            - cart_to_transaction_rate
        )

        if first_stage_loss >= second_stage_loss:

            st.warning(
                "Primary diagnostic priority: Product View → "
                "Add to Cart. This is currently the larger "
                "observed funnel loss."
            )

        else:

            st.warning(
                "Primary diagnostic priority: Add to Cart → "
                "Transaction. Users are progressing to cart "
                "more effectively than they are completing "
                "transactions."
            )


# ============================================================
# PAGE 2 — NEW VISITOR MONITOR
# ============================================================

elif page == "🆕 New Visitor Monitor":

    st.header(
        "New Visitor Monitor"
    )

    st.write(
        "Track users entering the observed product ecosystem "
        "for the first time during the selected period."
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "New Visitors",
        integer_format(
            new_visitor_count
        )
    )

    col2.metric(
        "Share of Visitors",
        percentage(
            new_visitor_share
        )
    )

    col3.metric(
        "New Visitor Events",
        integer_format(
            len(new_visitor_events)
        )
    )

    col4.metric(
        "New Visitor Transactions",
        integer_format(
            new_visitor_transactions
        )
    )

    st.divider()

    if new_visitors_in_period.empty:

        st.info(
            "No new visitors were identified for the selected "
            "period and user scope."
        )

    else:

        st.subheader(
            "New Visitor Arrival Trend"
        )

        arrivals = (
            new_visitors_in_period
            .groupby(
                "first_activity_date",
                sort=True
            )
            .size()
            .rename(
                "new_visitors"
            )
        )

        st.bar_chart(
            arrivals
        )

        st.divider()

        st.subheader(
            "New Visitor Engagement"
        )

        new_visitor_profile = (
            new_visitor_events
            .groupby(
                "visitorid",
                sort=False
            )
            .agg(
                first_activity=(
                    "date_day",
                    "min"
                ),
                events=(
                    "event",
                    "size"
                ),
                views=(
                    "is_view",
                    "sum"
                ),
                carts=(
                    "is_addtocart",
                    "sum"
                ),
                transactions=(
                    "is_transaction",
                    "sum"
                )
            )
            .reset_index()
            .sort_values(
                "events",
                ascending=False
            )
        )

        st.dataframe(
            new_visitor_profile,
            use_container_width=True,
            hide_index=True
        )

        csv_data = (
            new_visitor_profile
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ Download New Visitor Analysis",
            data=csv_data,
            file_name=(
                "new_visitor_analysis.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# PAGE 3 — FUNNEL INVESTIGATOR
# ============================================================

elif page == "🔻 Funnel Investigator":

    st.header(
        "Funnel Investigator"
    )

    st.write(
        "Configure the investigation and identify where "
        "users are dropping out of the observed funnel."
    )

    st.subheader(
        "🔍 Investigation Setup"
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        funnel_scope = st.selectbox(
            "Analyze",
            [
                "Event Funnel",
                "Unique Visitors"
            ]
        )

    with col2:

        analysis_focus = st.selectbox(
            "Investigation Focus",
            [
                "Largest Drop-Off",
                "Conversion Efficiency",
                "Stage Comparison"
            ]
        )

    run_funnel = st.button(
        "🔍 Run Funnel Analysis",
        use_container_width=True
    )

    if run_funnel:

        if funnel_scope == "Event Funnel":

            stage_values = [

                total_views,

                total_add_to_carts,

                total_transactions

            ]

        else:

            stage_values = [

                filtered_events.loc[
                    filtered_events[
                        "is_view"
                    ] == 1,
                    "visitorid"
                ].nunique(),

                filtered_events.loc[
                    filtered_events[
                        "is_addtocart"
                    ] == 1,
                    "visitorid"
                ].nunique(),

                filtered_events.loc[
                    filtered_events[
                        "is_transaction"
                    ] == 1,
                    "visitorid"
                ].nunique()

            ]

        funnel_data = (
            pd.DataFrame(
                {
                    "Funnel Stage": [
                        "Product View",
                        "Add to Cart",
                        "Transaction"
                    ],
                    "Value": stage_values
                }
            )
            .set_index(
                "Funnel Stage"
            )
        )

        st.divider()

        st.subheader(
            "Funnel Results"
        )

        st.bar_chart(
            funnel_data
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "Views",
            integer_format(
                stage_values[0]
            )
        )

        col2.metric(
            "Add to Cart",
            integer_format(
                stage_values[1]
            )
        )

        col3.metric(
            "Transactions",
            integer_format(
                stage_values[2]
            )
        )

        first_transition = (
            safe_divide(
                stage_values[1],
                stage_values[0]
            )
        )

        second_transition = (
            safe_divide(
                stage_values[2],
                stage_values[1]
            )
        )

        st.subheader(
            "Investigation Finding"
        )

        if stage_values[0] == 0:

            st.info(
                "No funnel activity is available for this selection."
            )

        elif analysis_focus == "Largest Drop-Off":

            if first_transition <= second_transition:

                st.error(
                    "Primary investigation target: Product "
                    "View → Add to Cart. Only "
                    f"{percentage(first_transition)} progress "
                    "through this transition."
                )

            else:

                st.error(
                    "Primary investigation target: Add to "
                    "Cart → Transaction. Only "
                    f"{percentage(second_transition)} progress "
                    "through this transition."
                )

        elif analysis_focus == "Conversion Efficiency":

            st.info(
                f"View → Cart efficiency: "
                f"{percentage(first_transition)} | "
                f"Cart → Transaction efficiency: "
                f"{percentage(second_transition)}"
            )

        else:

            comparison = pd.DataFrame(
                {
                    "Stage Transition": [
                        "View → Cart",
                        "Cart → Transaction"
                    ],
                    "Conversion Rate": [
                        first_transition,
                        second_transition
                    ]
                }
            )

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "Configure the investigation and click "
            "'Run Funnel Analysis'."
        )


# ============================================================
# PAGE 4 — RETENTION & COHORTS
# ============================================================

elif page == "🔄 Retention & Cohorts":

    st.header(
        "Retention & Cohorts"
    )

    st.write(
        "Investigate repeat activity patterns among the "
        "selected users."
    )

    analysis_type = st.selectbox(
        "Retention Analysis",
        [
            "Returning Visitor Analysis",
            "Visitor Activity Distribution"
        ]
    )

    run_retention = st.button(
        "🔄 Analyze Retention",
        use_container_width=True
    )

    if run_retention:

        if filtered_events.empty:

            st.info(
                "No data is available for the selected filters."
            )

        else:

            visitor_activity = (
                filtered_events
                .groupby(
                    "visitorid",
                    sort=False
                )
                .agg(
                    first_activity=(
                        "date_day",
                        "min"
                    ),
                    last_activity=(
                        "date_day",
                        "max"
                    ),
                    active_days=(
                        "date_day",
                        "nunique"
                    ),
                    total_events=(
                        "event",
                        "size"
                    )
                )
                .reset_index()
            )

            returning_visitors_count = int(
                (
                    visitor_activity[
                        "active_days"
                    ] > 1
                )
                .sum()
            )

            returning_rate = (
                safe_divide(
                    returning_visitors_count,
                    len(visitor_activity)
                )
            )

            if analysis_type == "Returning Visitor Analysis":

                col1, col2, col3 = (
                    st.columns(3)
                )

                col1.metric(
                    "Visitors",
                    integer_format(
                        len(visitor_activity)
                    )
                )

                col2.metric(
                    "Returning Visitors",
                    integer_format(
                        returning_visitors_count
                    )
                )

                col3.metric(
                    "Returning Rate",
                    percentage(
                        returning_rate
                    )
                )

                st.caption(
                    "Returning Visitors are users active on more "
                    "than one observed calendar day within the "
                    "selected analysis period."
                )

            else:

                distribution = (
                    visitor_activity[
                        "total_events"
                    ]
                    .value_counts()
                    .sort_index()
                    .head(30)
                )

                st.bar_chart(
                    distribution
                )


# ============================================================
# PAGE 5 — USER SEGMENTATION
# ============================================================

elif page == "👥 User Segmentation":

    try:

        st.header(
            "User Segmentation Explorer"
        )

        st.write(
            "Investigate mutually exclusive behavioral segments "
            "created from the complete observation history."
        )

        # ----------------------------------------------------
        # DEBUG CHECKS
        # ----------------------------------------------------

        st.caption(
            f"Available segments: {len(segment_options) - 1}"
        )

        segment_to_analyze = st.selectbox(
            "Select Segment",
            segment_options[1:]
        )

        run_segment = st.button(
            "👥 Analyze Segment",
            use_container_width=True
        )

        if run_segment:

            # ------------------------------------------------
            # FIND VISITORS IN SELECTED SEGMENT
            # ------------------------------------------------

            segment_features = (
                visitor_segments.loc[
                    visitor_segments["segment"]
                    == segment_to_analyze
                ].copy()
            )

            if segment_features.empty:

                st.warning(
                    f"No visitors found in segment: "
                    f"{segment_to_analyze}"
                )

                st.stop()

            segment_visitors = set(
                segment_features["visitorid"]
            )

            # ------------------------------------------------
            # FILTER EVENTS
            # ------------------------------------------------

            segment_events = filter_events(
                events,
                start_date,
                end_date,
                segment_visitors
            )

            # ------------------------------------------------
            # CALCULATE METRICS
            # ------------------------------------------------

            segment_metrics = calculate_period_metrics(
                segment_events
            )

            # ------------------------------------------------
            # KPI METRICS
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Segment Visitors",
                integer_format(
                    len(segment_features)
                )
            )

            col2.metric(
                "Events in Period",
                integer_format(
                    segment_metrics.get("events", 0)
                )
            )

            col3.metric(
                "Transactions",
                integer_format(
                    segment_metrics.get(
                        "transactions",
                        0
                    )
                )
            )

            col4.metric(
                "View → Transaction",
                percentage(
                    segment_metrics.get(
                        "overall_conversion",
                        0
                    )
                )
            )

            st.divider()

            # ------------------------------------------------
            # SEGMENT PROFILE
            # ------------------------------------------------

            st.subheader(
                "Segment Profile"
            )

            profile_columns = [
                "active_days",
                "total_events",
                "view_events",
                "cart_events",
                "transaction_events",
                "recency_days"
            ]

            missing_columns = [
                column
                for column in profile_columns
                if column not in segment_features.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

                st.write(
                    "Available columns:"
                )

                st.write(
                    list(segment_features.columns)
                )

                st.stop()

            profile = pd.DataFrame(
                {
                    "Metric": [

                        "Average Active Days",

                        "Average Total Events",

                        "Average View Events",

                        "Average Cart Events",

                        "Average Transaction Events",

                        "Average Recency Days"

                    ],

                    "Value": [

                        round(
                            segment_features[
                                "active_days"
                            ].mean(),
                            2
                        ),

                        round(
                            segment_features[
                                "total_events"
                            ].mean(),
                            2
                        ),

                        round(
                            segment_features[
                                "view_events"
                            ].mean(),
                            2
                        ),

                        round(
                            segment_features[
                                "cart_events"
                            ].mean(),
                            2
                        ),

                        round(
                            segment_features[
                                "transaction_events"
                            ].mean(),
                            2
                        ),

                        round(
                            segment_features[
                                "recency_days"
                            ].mean(),
                            2
                        )

                    ]
                }
            )

            st.dataframe(
                profile,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # ------------------------------------------------
            # SEGMENTATION RULEBOOK
            # ------------------------------------------------

            st.subheader(
                "Segmentation Rulebook"
            )

            rulebook = pd.DataFrame(
                {
                    "Segment": [

                        "Repeat Buyers",

                        "One-Time Buyers",

                        "High-Intent Visitors",

                        "Cart Abandoners",

                        "At-Risk Visitors",

                        "New Visitors",

                        "Highly Engaged Browsers",

                        "Other Visitors"

                    ],

                    "Rule": [

                        "At least 2 unique transactions",

                        "Exactly 1 unique transaction",

                        "No transaction and at least "
                        f"{segment_thresholds['high_intent_cart_threshold']} "
                        "cart events",

                        "Cart activity below the high-intent threshold "
                        "and no transaction",

                        "No transaction, activity on multiple days, "
                        "and at least 14 days of inactivity",

                        "First observed activity from "
                        f"{segment_thresholds['new_visitor_cutoff'].date()} "
                        "to the end of the observation window",

                        "No cart or transaction activity, at least "
                        f"{segment_thresholds['active_day_threshold']} "
                        "active days, and at least "
                        f"{segment_thresholds['view_event_threshold']} "
                        "view events",

                        "Remaining visitors"

                    ]
                }
            )

            st.dataframe(
                rulebook,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Select a behavioral segment and click "
                "'Analyze Segment'."
            )

    except Exception as e:

        st.error(
            "User Segmentation encountered an error."
        )

        st.exception(e)

        with st.expander(
            "Debug Information"
        ):

            st.write(
                "Selected segment:",
                segment_to_analyze
                if "segment_to_analyze" in locals()
                else "Not available"
            )

            st.write(
                "visitor_segments columns:"
            )

            if "visitor_segments" in globals():

                st.write(
                    list(visitor_segments.columns)
                )

            st.write(
                "Events columns:"
            )

            if "events" in globals():

                st.write(
                    list(events.columns)
                )


# ============================================================
# PAGE 6 — PRODUCT & CATEGORY INTELLIGENCE
# ============================================================

elif page == "📦 Product & Category Intelligence":

    st.header(
        "Product & Category Intelligence"
    )

    st.write(
        "Explore product engagement and identify items "
        "with strong interest but weak downstream progression."
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        analysis_metric = st.selectbox(
            "Rank Products By",
            [
                "Total Engagement",
                "Views",
                "Add-to-Cart Activity",
                "Transactions"
            ]
        )

    with col2:

        top_n = st.slider(
            "Number of Products",
            min_value=5,
            max_value=30,
            value=10
        )

    run_product_analysis = st.button(
        "📦 Analyze Products",
        use_container_width=True
    )

    if run_product_analysis:

        filtered_events_enriched = (
            filter_events(
                events_enriched,
                start_date,
                end_date,
                selected_visitors
            )
        )

        if filtered_events_enriched.empty:

            st.info(
                "No product activity is available for the "
                "selected filters."
            )

        else:

            product_metrics = (
                filtered_events_enriched
                .groupby(
                    "itemid",
                    sort=False
                )
                .agg(
                    total_events=(
                        "event",
                        "size"
                    ),
                    visitors=(
                        "visitorid",
                        "nunique"
                    ),
                    views=(
                        "is_view",
                        "sum"
                    ),
                    add_to_carts=(
                        "is_addtocart",
                        "sum"
                    ),
                    transactions=(
                        "is_transaction",
                        "sum"
                    )
                )
                .reset_index()
            )

            product_metrics = (
                add_conversion_rates(
                    product_metrics
                )
            )

            ranking_map = {

                "Total Engagement": (
                    "total_events"
                ),

                "Views": (
                    "views"
                ),

                "Add-to-Cart Activity": (
                    "add_to_carts"
                ),

                "Transactions": (
                    "transactions"
                )

            }

            ranking_column = (
                ranking_map[
                    analysis_metric
                ]
            )

            top_products = (
                product_metrics
                .sort_values(
                    ranking_column,
                    ascending=False
                )
                .head(top_n)
            )

            st.subheader(
                f"Top {top_n} Products"
            )

            st.dataframe(
                top_products,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader(
                "High-Interest / Low-Progression Diagnostic"
            )

            minimum_views = (
                product_metrics[
                    "views"
                ]
                .quantile(0.75)
            )

            minimum_conversion = (
                product_metrics[
                    "view_to_cart_rate"
                ]
                .median()
            )

            opportunities = (
                product_metrics.loc[
                    (
                        product_metrics[
                            "views"
                        ]
                        >= minimum_views
                    )
                    &
                    (
                        product_metrics[
                            "view_to_cart_rate"
                        ]
                        < minimum_conversion
                    )
                ]
                .sort_values(
                    [
                        "views",
                        "view_to_cart_rate"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                )
                .head(top_n)
            )

            if opportunities.empty:

                st.info(
                    "No strong high-interest / low-progression "
                    "product signal was identified using the "
                    "current data-driven thresholds."
                )

            else:

                st.dataframe(
                    opportunities,
                    use_container_width=True,
                    hide_index=True
                )

            category_column = (
                get_category_column(
                    filtered_events_enriched
                )
            )

            if category_column is not None:

                st.divider()

                st.subheader(
                    "Category Intelligence"
                )

                category_metrics = (
                    filtered_events_enriched
                    .dropna(
                        subset=[
                            category_column
                        ]
                    )
                    .groupby(
                        category_column,
                        sort=False
                    )
                    .agg(
                        visitors=(
                            "visitorid",
                            "nunique"
                        ),
                        events=(
                            "event",
                            "size"
                        ),
                        views=(
                            "is_view",
                            "sum"
                        ),
                        add_to_carts=(
                            "is_addtocart",
                            "sum"
                        ),
                        transactions=(
                            "is_transaction",
                            "sum"
                        )
                    )
                    .reset_index()
                )

                category_metrics = (
                    add_conversion_rates(
                        category_metrics
                    )
                )

                category_metrics = (
                    category_metrics
                    .sort_values(
                        "visitors",
                        ascending=False
                    )
                    .head(top_n)
                )

                st.dataframe(
                    category_metrics,
                    use_container_width=True,
                    hide_index=True
                )

            csv_data = (
                product_metrics
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "⬇️ Download Product Intelligence",
                data=csv_data,
                file_name=(
                    "product_intelligence.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

    else:

        st.info(
            "Choose the ranking options and click "
            "'Analyze Products'."
        )


# ============================================================
# PAGE 7 — GROWTH DECISION LAB
# ============================================================

elif page == "🧪 Growth Decision Lab":

    st.header(
        "Growth Decision Lab"
    )

    st.write(
        "Test transparent what-if scenarios using the observed "
        "dataset and explicit assumptions."
    )

    scenario = st.selectbox(
        "Choose Metric to Improve",
        [
            "View to Cart Conversion",
            "Cart to Transaction Conversion",
            "New Visitor Growth"
        ]
    )

    if scenario == "View to Cart Conversion":

        current_rate = (
            view_to_cart_rate
        )

        target_rate = st.slider(
            "Target View → Cart Rate",
            min_value=float(
                current_rate
            ),
            max_value=1.0,
            value=min(
                float(
                    current_rate + 0.05
                ),
                1.0
            ),
            step=0.005,
            format="%.3f"
        )

        run_simulation = st.button(
            "🧪 Simulate Growth Impact",
            use_container_width=True
        )

        if run_simulation:

            projected_carts = (
                total_views
                * target_rate
            )

            projected_transactions = (
                projected_carts
                * cart_to_transaction_rate
            )

            additional_transactions = max(
                0,
                projected_transactions
                - total_transactions
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            col1.metric(
                "Current Rate",
                percentage(
                    current_rate
                )
            )

            col2.metric(
                "Target Rate",
                percentage(
                    target_rate
                )
            )

            col3.metric(
                "Estimated Additional Transactions",
                integer_format(
                    additional_transactions
                )
            )

            st.info(
                "Scenario estimate only. The calculation assumes "
                "the current Cart → Transaction conversion rate "
                "remains unchanged."
            )

    elif scenario == "Cart to Transaction Conversion":

        current_rate = (
            cart_to_transaction_rate
        )

        target_rate = st.slider(
            "Target Cart → Transaction Rate",
            min_value=float(
                current_rate
            ),
            max_value=1.0,
            value=min(
                float(
                    current_rate + 0.05
                ),
                1.0
            ),
            step=0.005,
            format="%.3f"
        )

        run_simulation = st.button(
            "🧪 Simulate Growth Impact",
            use_container_width=True
        )

        if run_simulation:

            projected_transactions = (
                total_add_to_carts
                * target_rate
            )

            additional_transactions = max(
                0,
                projected_transactions
                - total_transactions
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            col1.metric(
                "Current Rate",
                percentage(
                    current_rate
                )
            )

            col2.metric(
                "Target Rate",
                percentage(
                    target_rate
                )
            )

            col3.metric(
                "Estimated Additional Transactions",
                integer_format(
                    additional_transactions
                )
            )

            st.info(
                "Scenario estimate only. The calculation assumes "
                "the number of Add-to-Cart events remains unchanged."
            )

    else:

        growth_percentage = st.slider(
            "Expected Increase in New Visitors (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5
        )

        run_simulation = st.button(
            "🧪 Simulate Acquisition Impact",
            use_container_width=True
        )

        if run_simulation:

            projected_new_visitors = (
                new_visitor_count
                * (
                    1
                    + growth_percentage / 100
                )
            )

            additional_new_visitors = (
                projected_new_visitors
                - new_visitor_count
            )

            projected_transactions = (
                additional_new_visitors
                * new_visitor_conversion
            )

            col1, col2, col3 = (
                st.columns(3)
            )

            col1.metric(
                "Current New Visitors",
                integer_format(
                    new_visitor_count
                )
            )

            col2.metric(
                "Projected New Visitors",
                integer_format(
                    projected_new_visitors
                )
            )

            col3.metric(
                "Estimated Additional Transactions",
                integer_format(
                    projected_transactions
                )
            )

            st.info(
                "Scenario estimate only. This assumes newly "
                "acquired visitors behave similarly to the "
                "observed new visitors in the selected period."
            )


# ============================================================
# PAGE 8 — GROWTH OPPORTUNITIES
# ============================================================

elif page == "🎯 Growth Opportunities":

    st.header(
        "Growth Opportunity Engine"
    )

    st.write(
        "Automatically identify and prioritize observed "
        "growth opportunities from acquisition, conversion, "
        "engagement, and product behavior."
    )

    focus_areas = st.multiselect(
        "Focus Areas",
        [
            "Acquisition",
            "Conversion",
            "Engagement",
            "Product Performance"
        ],
        default=[
            "Acquisition",
            "Conversion",
            "Engagement",
            "Product Performance"
        ]
    )

    run_opportunities = st.button(
        "🎯 Find Growth Opportunities",
        use_container_width=True
    )

    if run_opportunities:

        opportunities = []


        # ========================================================
        # ACQUISITION
        # ========================================================

        if "Acquisition" in focus_areas:

            if new_visitor_share < 0.20:

                opportunities.append(
                    {
                        "Priority": "Medium",

                        "Area": "Acquisition",

                        "Problem": (
                            "Low new visitor contribution"
                        ),

                        "Evidence": (
                            "New visitors represent only "
                            f"{percentage(new_visitor_share)} "
                            "of selected visitors."
                        ),

                        "Recommended Action": (
                            "Investigate how the product can "
                            "attract and activate more first-time "
                            "visitors."
                        )
                    }
                )

            else:

                opportunities.append(
                    {
                        "Priority": "Medium",

                        "Area": "Acquisition",

                        "Problem": (
                            "New visitor activation opportunity"
                        ),

                        "Evidence": (
                            f"{integer_format(new_visitor_count)} "
                            "new visitors entered during the "
                            "selected period."
                        ),

                        "Recommended Action": (
                            "Focus on onboarding and first-visit "
                            "progression to convert new traffic "
                            "into deeper engagement."
                        )
                    }
                )


        # ========================================================
        # CONVERSION
        # ========================================================

        if "Conversion" in focus_areas:

            if (
                view_to_cart_rate
                <= cart_to_transaction_rate
            ):

                opportunities.append(
                    {
                        "Priority": "High",

                        "Area": "Conversion",

                        "Problem": (
                            "Low View → Cart progression"
                        ),

                        "Evidence": (
                            "Only "
                            f"{percentage(view_to_cart_rate)} "
                            "of product views progress to cart."
                        ),

                        "Recommended Action": (
                            "Investigate product pages and "
                            "high-view, low-cart products."
                        )
                    }
                )

            else:

                opportunities.append(
                    {
                        "Priority": "High",

                        "Area": "Conversion",

                        "Problem": (
                            "Low Cart → Transaction progression"
                        ),

                        "Evidence": (
                            "Only "
                            f"{percentage(cart_to_transaction_rate)} "
                            "of cart events progress to transactions."
                        ),

                        "Recommended Action": (
                            "Investigate purchase completion "
                            "behavior and downstream friction."
                        )
                    }
                )


        # ========================================================
        # ENGAGEMENT
        # ========================================================

        if "Engagement" in focus_areas:

            high_intent_count = int(
                (
                    visitor_segments[
                        "segment"
                    ]
                    == "High-Intent Visitors"
                )
                .sum()
            )

            cart_abandoner_count = int(
                (
                    visitor_segments[
                        "segment"
                    ]
                    == "Cart Abandoners"
                )
                .sum()
            )

            at_risk_count = int(
                (
                    visitor_segments[
                        "segment"
                    ]
                    == "At-Risk Visitors"
                )
                .sum()
            )

            if (
                high_intent_count
                + cart_abandoner_count
                > 0
            ):

                opportunities.append(
                    {
                        "Priority": "High",

                        "Area": "Engagement",

                        "Problem": (
                            "Users demonstrate purchase intent "
                            "without conversion"
                        ),

                        "Evidence": (
                            f"{integer_format(high_intent_count)} "
                            "high-intent visitors and "
                            f"{integer_format(cart_abandoner_count)} "
                            "cart abandoners were identified."
                        ),

                        "Recommended Action": (
                            "Prioritize investigation of "
                            "high-intent and cart-abandoning "
                            "behavior."
                        )
                    }
                )

            if at_risk_count > 0:

                opportunities.append(
                    {
                        "Priority": "Medium",

                        "Area": "Engagement",

                        "Problem": (
                            "Previously active users show "
                            "prolonged inactivity"
                        ),

                        "Evidence": (
                            f"{integer_format(at_risk_count)} "
                            "visitors meet the at-risk "
                            "behavioral rule."
                        ),

                        "Recommended Action": (
                            "Investigate re-engagement "
                            "opportunities for inactive "
                            "previously active visitors."
                        )
                    }
                )


        # ========================================================
        # PRODUCT PERFORMANCE
        # ========================================================

        if "Product Performance" in focus_areas:

            filtered_events_enriched = (
                filter_events(
                    events_enriched,
                    start_date,
                    end_date,
                    selected_visitors
                )
            )

            if not filtered_events_enriched.empty:

                product_metrics = (
                    filtered_events_enriched
                    .groupby(
                        "itemid",
                        sort=False
                    )
                    .agg(
                        views=(
                            "is_view",
                            "sum"
                        ),
                        add_to_carts=(
                            "is_addtocart",
                            "sum"
                        ),
                        transactions=(
                            "is_transaction",
                            "sum"
                        )
                    )
                    .reset_index()
                )

                product_metrics = (
                    add_conversion_rates(
                        product_metrics
                    )
                )

                high_view_threshold = (
                    product_metrics[
                        "views"
                    ]
                    .quantile(0.75)
                )

                low_conversion_threshold = (
                    product_metrics[
                        "view_to_cart_rate"
                    ]
                    .median()
                )

                weak_progression_products = (
                    product_metrics.loc[
                        (
                            product_metrics[
                                "views"
                            ]
                            >= high_view_threshold
                        )
                        &
                        (
                            product_metrics[
                                "view_to_cart_rate"
                            ]
                            < low_conversion_threshold
                        )
                    ]
                )

                if not weak_progression_products.empty:

                    opportunities.append(
                        {
                            "Priority": "High",

                            "Area": (
                                "Product Performance"
                            ),

                            "Problem": (
                                "High-interest products with "
                                "weak cart progression"
                            ),

                            "Evidence": (
                                f"{integer_format(len(weak_progression_products))} "
                                "products meet the diagnostic "
                                "criteria."
                            ),

                            "Recommended Action": (
                                "Investigate these products first "
                                "for possible downstream funnel "
                                "friction."
                            )
                        }
                    )


        # ========================================================
        # FINAL OPPORTUNITY OUTPUT
        # ========================================================

        opportunities_df = pd.DataFrame(
            opportunities
        )

        if opportunities_df.empty:

            st.info(
                "No opportunity signals were generated for "
                "the selected focus areas."
            )

        else:

            priority_order = {
                "High": 1,
                "Medium": 2,
                "Low": 3
            }

            opportunities_df[
                "priority_rank"
            ] = (
                opportunities_df[
                    "Priority"
                ]
                .map(
                    priority_order
                )
            )

            opportunities_df = (
                opportunities_df
                .sort_values(
                    "priority_rank"
                )
                .drop(
                    columns="priority_rank"
                )
                .reset_index(
                    drop=True
                )
            )

            st.dataframe(
                opportunities_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader(
                "🏆 Recommended Starting Point"
            )

            top_opportunity = (
                opportunities_df.iloc[0]
            )

            st.success(
                f"Start with: "
                f"{top_opportunity['Problem']}"
            )

            st.write(
                f"**Area:** "
                f"{top_opportunity['Area']}"
            )

            st.write(
                f"**Why:** "
                f"{top_opportunity['Evidence']}"
            )

            st.write(
                f"**Next action:** "
                f"{top_opportunity['Recommended Action']}"
            )

            csv_data = (
                opportunities_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "⬇️ Download Opportunity Report",
                data=csv_data,
                file_name=(
                    "growth_opportunity_report.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

    else:

        st.info(
            "Select focus areas and click "
            "'Find Growth Opportunities'."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Product & Growth Intelligence Platform | "
    "Interactive Data Science Decision-Support Application"
)

st.caption(
    "Behavioral segments are descriptive analytical constructs "
    "based on observed event history. Opportunity signals are "
    "diagnostic priorities and do not establish causality."
)