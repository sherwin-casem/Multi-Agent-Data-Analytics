import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np # Added numpy import
from typing import Dict, Any, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChartGenerator:
    """
    Utility class for generating various types of charts and visualizations.
    """
    
    def __init__(self):
        # Color scheme matching the application theme
        self.color_palette = [
            '#2E86AB',  # Primary blue
            '#A23B72',  # Secondary purple
            '#F18F01',  # Accent orange
            '#28A745',  # Success green
            '#DC3545',  # Error red
            '#6F42C1',  # Purple
            '#20C997',  # Teal
            '#FD7E14'   # Orange
        ]
        
        self.layout_theme = {
            'plot_bgcolor': '#F8F9FA',
            'paper_bgcolor': '#F8F9FA',
            'font_color': '#212529',
        }
        self.grid_color = '#E9ECEF' # Store grid color separately for axis application
        logging.info("ChartGenerator initialized with default color palette and theme.")
    
    def create_chart(self, data: Union[pd.DataFrame, str], chart_config: Dict[str, Any]) -> go.Figure:
        """
        Create a chart based on configuration.
        
        Args:
            data (Union[pd.DataFrame, str]): DataFrame containing the data or unstructured text.
            chart_config (Dict[str, Any]): Configuration dictionary specifying chart type and parameters.
            
        Returns:
            go.Figure: Plotly figure object. Returns an error chart if input data is not a DataFrame
                       or if chart creation fails.
        """
        if not isinstance(data, pd.DataFrame):
            error_message = f"Chart generation requires structured data (Pandas DataFrame), but received {type(data)}. Charts are not applicable for unstructured text."
            logging.warning(error_message)
            return self._create_error_chart(error_message)

        if data.empty:
            error_message = "Cannot create chart from an empty DataFrame."
            logging.warning(error_message)
            return self._create_error_chart(error_message)

        try:
            chart_type = chart_config.get('type', 'bar')
            logging.info(f"Attempting to create chart of type: {chart_type}")
            
            # Ensure columns exist in data before attempting to create chart
            # This check is now also in VisualizationAgent.analyze_query, reinforcing robustness
            required_cols = []
            if chart_type in ["bar", "line", "scatter", "area", "box", "violin", "histogram"]:
                if 'x_column' in chart_config and chart_config['x_column']:
                    required_cols.append(chart_config['x_column'])
                if 'y_column' in chart_config and chart_config['y_column']:
                    required_cols.append(chart_config['y_column'])
            elif chart_type == 'pie' and 'names_column' in chart_config and chart_config['names_column']:
                required_cols.append(chart_config['names_column'])
            elif chart_type == 'sunburst' and 'path_columns' in chart_config and chart_config['path_columns']:
                required_cols.extend(chart_config['path_columns'])

            # Check if all required columns are in the DataFrame for the selected chart type
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                error_message = f"Missing required columns for {chart_type} chart: {', '.join(missing_cols)}. Available columns: {', '.join(data.columns.tolist())}"
                logging.error(error_message)
                return self._create_error_chart(error_message)

            if chart_type == 'bar':
                fig = self.create_bar_chart(data, chart_config)
            elif chart_type == 'line':
                fig = self.create_line_chart(data, chart_config)
            elif chart_type == 'scatter':
                fig = self.create_scatter_plot(data, chart_config)
            elif chart_type == 'pie':
                fig = self.create_pie_chart(data, chart_config)
            elif chart_type == 'histogram':
                fig = self.create_histogram(data, chart_config)
            elif chart_type == 'box':
                fig = self.create_box_plot(data, chart_config)
            elif chart_type == 'heatmap':
                fig = self.create_heatmap(data, chart_config)
            elif chart_type == 'area':
                fig = self.create_area_chart(data, chart_config)
            elif chart_type == 'violin':
                fig = self.create_violin_plot(data, chart_config)
            elif chart_type == 'sunburst':
                fig = self.create_sunburst_chart(data, chart_config)
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            self._apply_theme(fig) # Apply theme after chart creation
            return fig
                
        except Exception as e:
            logging.error(f"Critical error during chart creation for type {chart_config.get('type', 'N/A')}: {e}", exc_info=True)
            return self._create_error_chart(f"Error creating chart: {str(e)}")
    
    def create_bar_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a bar chart"""
        logging.info("Creating bar chart.")
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        title = config.get('title', 'Bar Chart')
        orientation = config.get('orientation', 'vertical')
        
        if orientation == 'horizontal':
            fig = px.bar(data, y=x_col, x=y_col, color=color_col, 
                         title=title, color_discrete_sequence=self.color_palette,
                         orientation='h')
        else:
            fig = px.bar(data, x=x_col, y=y_col, color=color_col,
                         title=title, color_discrete_sequence=self.color_palette)
        
        return fig
    
    def create_line_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a line chart"""
        logging.info("Creating line chart.")
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        title = config.get('title', 'Line Chart')
        
        fig = px.line(data, x=x_col, y=y_col, color=color_col,
                      title=title, color_discrete_sequence=self.color_palette)
        
        # Add markers
        fig.update_traces(mode='lines+markers')
        
        return fig
    
    def create_scatter_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a scatter plot"""
        logging.info("Creating scatter plot.")
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        size_col = config.get('size_column')
        title = config.get('title', 'Scatter Plot')
        
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col, size=size_col,
                         title=title, color_discrete_sequence=self.color_palette)
        
        # Add trendline if requested
        if config.get('trendline') and x_col and y_col: # Ensure columns exist for trendline
            if pd.api.types.is_numeric_dtype(data[x_col]) and pd.api.types.is_numeric_dtype(data[y_col]):
                trend_data = data[[x_col, y_col]].dropna()
                if not trend_data.empty:
                    z = np.polyfit(trend_data[x_col], trend_data[y_col], 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(x=trend_data[x_col], y=p(trend_data[x_col]), mode='lines', 
                                             name='Trendline', line=dict(color='red', dash='dash')))
            else:
                logging.warning("Trendline requested but x_column or y_column are not numeric.")
        
        return fig
    
    def create_pie_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a pie chart"""
        logging.info("Creating pie chart.")
        values_col = config.get('values_column')
        names_col = config.get('names_column')
        title = config.get('title', 'Pie Chart')
        
        # If values column is not specified, use value counts
        if not values_col and names_col:
            if names_col in data.columns and not data[names_col].empty:
                pie_data = data[names_col].value_counts().reset_index()
                pie_data.columns = ['names', 'values'] 
                fig = px.pie(pie_data, values='values', names='names', title=title,
                             color_discrete_sequence=self.color_palette)
            else:
                logging.error(f"Names column '{names_col}' not found or is empty for pie chart.")
                return self._create_error_chart(f"Cannot create pie chart: '{names_col}' column is missing or empty.")
        elif values_col and names_col:
            fig = px.pie(data, values=values_col, names=names_col, title=title,
                         color_discrete_sequence=self.color_palette)
        else:
            logging.error("Insufficient columns provided for pie chart (need names_column and optionally values_column).")
            return self._create_error_chart("Insufficient columns for pie chart.")
        
        return fig
    
    def create_histogram(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a histogram"""
        logging.info("Creating histogram.")
        x_col = config.get('x_column')
        color_col = config.get('color_column')
        title = config.get('title', 'Histogram')
        bins = config.get('bins', 30)
        
        # Ensure x_col is numeric for a histogram
        if x_col not in data.columns or not pd.api.types.is_numeric_dtype(data[x_col]):
            logging.error(f"Cannot create histogram: x_column '{x_col}' is not numeric or missing.")
            return self._create_error_chart(f"Histogram requires a numeric column for X-axis. '{x_col}' is not suitable.")

        fig = px.histogram(data, x=x_col, color=color_col, title=title,
                           nbins=bins, color_discrete_sequence=self.color_palette)
        
        return fig
    
    def create_box_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a box plot"""
        logging.info("Creating box plot.")
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        title = config.get('title', 'Box Plot')
        
        # Ensure y_col is numeric for a box plot (and x_col can be categorical)
        if y_col not in data.columns or not pd.api.types.is_numeric_dtype(data[y_col]):
            logging.error(f"Cannot create box plot: y_column '{y_col}' is not numeric or missing.")
            return self._create_error_chart(f"Box plot requires a numeric column for Y-axis. '{y_col}' is not suitable.")

        fig = px.box(data, x=x_col, y=y_col, color=color_col, title=title,
                     color_discrete_sequence=self.color_palette)
        
        return fig
    
    def create_heatmap(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a heatmap (typically of correlation matrix)"""
        logging.info("Creating heatmap.")
        title = config.get('title', 'Heatmap')
        
        # Create correlation matrix for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            logging.error("Heatmap requires at least 2 numeric columns for correlation matrix.")
            return self._create_error_chart("Heatmap requires at least 2 numeric columns")
        
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, title=title, color_continuous_scale='RdBu_r',
                        aspect='auto', text_auto=True) 
        
        return fig
    
    def create_area_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create an area chart"""
        logging.info("Creating area chart.")
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        title = config.get('title', 'Area Chart')
        
        if not x_col or not y_col or x_col not in data.columns or y_col not in data.columns:
            logging.error(f"Missing x_column ('{x_col}') or y_column ('{y_col}') for area chart.")
            return self._create_error_chart("Area chart requires X and Y columns.")
        if not pd.api.types.is_numeric_dtype(data[y_col]):
            logging.error(f"Cannot create area chart: y_column '{y_col}' is not numeric.")
            return self._create_error_chart(f"Area chart requires a numeric column for Y-axis. '{y_col}' is not suitable.")

        fig = px.area(data, x=x_col, y=y_col, color=color_col, title=title,
                      color_discrete_sequence=self.color_palette)
        
        return fig
    
    def create_violin_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a violin plot"""
        logging.info("Creating violin plot.")
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        title = config.get('title', 'Violin Plot')
        
        # Ensure y_col is numeric for a violin plot
        if y_col not in data.columns or not pd.api.types.is_numeric_dtype(data[y_col]):
            logging.error(f"Cannot create violin plot: y_column '{y_col}' is not numeric or missing.")
            return self._create_error_chart(f"Violin plot requires a numeric column for Y-axis. '{y_col}' is not suitable.")

        fig = px.violin(data, x=x_col, y=y_col, color=color_col, title=title,
                        color_discrete_sequence=self.color_palette)
        
        return fig
    
    def create_sunburst_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a sunburst chart"""
        logging.info("Creating sunburst chart.")
        path_cols = config.get('path_columns', [])
        values_col = config.get('values_column')
        title = config.get('title', 'Sunburst Chart')
        
        if not path_cols:
            logging.error("Sunburst chart requires path columns to be specified.")
            return self._create_error_chart("Sunburst chart requires path columns")
        
        # Check if all path columns exist in data
        missing_path_cols = [col for col in path_cols if col not in data.columns]
        if missing_path_cols:
            logging.error(f"Missing path columns for sunburst chart: {', '.join(missing_path_cols)}.")
            return self._create_error_chart(f"Missing required path columns: {', '.join(missing_path_cols)}")
        
        # If values_col is provided, ensure it's numeric
        if values_col and (values_col not in data.columns or not pd.api.types.is_numeric_dtype(data[values_col])):
            logging.error(f"Values column '{values_col}' for sunburst chart is not numeric or missing.")
            return self._create_error_chart(f"Sunburst chart requires a numeric values column if specified. '{values_col}' is not suitable.")

        fig = px.sunburst(data, path=path_cols, values=values_col, title=title,
                          color_discrete_sequence=self.color_palette)
        
        return fig
    
    def create_multi_chart(self, data: pd.DataFrame, configs: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a multi-chart layout from a list of chart configurations.
        
        Args:
            data (pd.DataFrame): DataFrame containing the data.
            configs (List[Dict[str, Any]]): A list of chart configuration dictionaries.
            
        Returns:
            go.Figure: A Plotly figure containing multiple subplots.
        """
        logging.info(f"Creating multi-chart layout with {len(configs)} charts.")
        if not configs:
            logging.warning("No chart configurations provided for multi-chart layout.")
            return self._create_error_chart("No chart configurations provided")
        
        # Determine subplot layout
        n_charts = len(configs)
        if n_charts == 1:
            rows, cols = 1, 1
        elif n_charts == 2:
            rows, cols = 1, 2
        elif n_charts <= 4:
            rows, cols = 2, 2
        else:
            rows = int(np.ceil(n_charts / 3))
            cols = min(3, n_charts)
        
        # Create subplots
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[config.get('title', f'Chart {i+1}') for i, config in enumerate(configs)],
            horizontal_spacing=0.05, 
            vertical_spacing=0.1
        )
        
        # Add charts to subplots
        for i, config in enumerate(configs):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            try:
                # Create individual chart using the main create_chart method
                # Pass data and config. If create_chart returns an error chart, we can log it.
                individual_chart = self.create_chart(data, config)
                
                # Check if it's a valid chart or an error chart
                if individual_chart and hasattr(individual_chart, 'data') and individual_chart.data:
                     for trace in individual_chart.data:
                        fig.add_trace(trace, row=row, col=col)
                else:
                    logging.warning(f"Individual chart creation failed for config: {config.get('title', 'N/A')}. Adding placeholder.")
                    fig.add_annotation(
                        text=f"Chart Failed: {config.get('title', '')}",
                        xref=f"x{i+1} domain", yref=f"y{i+1} domain",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=10, color="red"),
                        row=row, col=col
                    )

            except Exception as e:
                logging.error(f"Error adding chart {config.get('title', 'N/A')} to subplot at ({row},{col}): {e}", exc_info=True)
                fig.add_annotation(
                    text=f"Error: {config.get('title', '')} failed. {str(e)[:50]}...",
                    xref=f"x{i+1} domain", yref=f"y{i+1} domain",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=10, color="red"),
                    row=row, col=col
                )
        
        # Update overall layout
        fig.update_layout(
            title_text="Multi-Chart Dashboard",
            title_x=0.5, 
            showlegend=True,
            **{k: v for k, v in self.layout_theme.items() if k not in ['gridcolor']} # Exclude gridcolor from direct layout
        )
        # Apply grid color to axes after layout is updated
        fig.update_xaxes(gridcolor=self.grid_color, zerolinecolor=self.grid_color)
        fig.update_yaxes(gridcolor=self.grid_color, zerolinecolor=self.grid_color)

        return fig
    
    def create_dashboard(self, data: pd.DataFrame, dashboard_config: Dict[str, Any] = None) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple suggested charts.

        Args:
            data (pd.DataFrame): The DataFrame to create the dashboard from.
            dashboard_config (Dict[str, Any], optional): Configuration for the dashboard (not heavily used for auto-gen).

        Returns:
            go.Figure: A Plotly figure representing the dashboard.
        """
        logging.info("Creating auto-generated dashboard.")
        if not isinstance(data, pd.DataFrame) or data.empty:
            logging.error("Dashboard creation requires a non-empty DataFrame.")
            return self._create_error_chart("Dashboard requires structured data (Pandas DataFrame).")

        charts_to_generate = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Add correlation heatmap if multiple numeric columns
        if len(numeric_cols) > 1:
            charts_to_generate.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix'
            })
        
        # Add distribution charts for numeric columns (up to 2 for brevity)
        for col in numeric_cols[:2]:
            charts_to_generate.append({
                'type': 'histogram',
                'x_column': col,
                'title': f'Distribution of {col}'
            })
        
        # Add bar charts for categorical vs numeric (up to 1 for brevity)
        if categorical_cols and numeric_cols:
            charts_to_generate.append({
                'type': 'bar',
                'x_column': categorical_cols[0],
                'y_column': numeric_cols[0],
                'title': f'{numeric_cols[0]} by {categorical_cols[0]}'
            })
        
        # Add a pie chart for a categorical column (if available)
        if categorical_cols:
            charts_to_generate.append({
                'type': 'pie',
                'names_column': categorical_cols[0],
                'title': f'Proportion of {categorical_cols[0]}'
            })

        if not charts_to_generate:
            logging.warning("No suitable charts could be auto-generated for the dashboard based on data types.")
            return self._create_error_chart("No suitable charts could be generated for the dashboard from the provided data.")

        # Create multi-chart layout
        return self.create_multi_chart(data, charts_to_generate)
    
    def _apply_theme(self, fig: go.Figure):
        """Apply consistent theme to charts"""
        
        fig.update_layout(
            title_font_size=18, 
            title_font_color=self.layout_theme['font_color'],
            margin=dict(l=40, r=40, t=80, b=40), 
            paper_bgcolor=self.layout_theme['paper_bgcolor'],
            plot_bgcolor=self.layout_theme['plot_bgcolor']
        )
        # Apply grid color to axes
        fig.update_xaxes(showline=True, linewidth=1, linecolor=self.grid_color, gridcolor=self.grid_color, zerolinecolor=self.grid_color)
        fig.update_yaxes(showline=True, linewidth=1, linecolor=self.grid_color, gridcolor=self.grid_color, zerolinecolor=self.grid_color)
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create an error chart when visualization fails"""
        logging.error(f"Creating error chart with message: {error_message}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"📊 Chart Error: <br>{error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=18, color="#DC3545", family="Inter"), 
            align="center",
            valign="middle"
        )
        
        fig.update_layout(
            title="Chart Generation Failed",
            title_font_size=20,
            **self.layout_theme,
            height=400 
        )
        return fig
    
    def get_chart_suggestions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get suggested chart configurations based on data characteristics.
        
        Args:
            data (pd.DataFrame): The DataFrame to get suggestions for.

        Returns:
            List[Dict[str, Any]]: A list of suggested chart configurations.
        """
        logging.info("Generating chart suggestions.")
        suggestions = []
        
        if data.empty:
            logging.warning("Cannot generate chart suggestions from an empty DataFrame.")
            return []

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Bar chart suggestions (e.g., top N categories by a numeric measure)
        if categorical_cols and numeric_cols:
            suggestions.append({
                'type': 'bar',
                'x_column': categorical_cols[0],
                'y_column': numeric_cols[0],
                'title': f'{numeric_cols[0]} by {categorical_cols[0]}',
                'description': 'Compare values across categories'
            })
        
        # Scatter plot suggestions
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'x_column': numeric_cols[0],
                'y_column': numeric_cols[1],
                'title': f'{numeric_cols[1]} vs {numeric_cols[0]}',
                'description': 'Explore relationship between two variables'
            })
        
        # Histogram suggestions
        for col in numeric_cols[:2]: 
            suggestions.append({
                'type': 'histogram',
                'x_column': col,
                'title': f'Distribution of {col}',
                'description': 'Understand data distribution'
            })
        
        # Pie chart suggestions
        for col in categorical_cols[:2]: 
            suggestions.append({
                'type': 'pie',
                'names_column': col,
                'title': f'{col} Distribution',
                'description': 'Show category proportions'
            })
        
        # Correlation heatmap
        if len(numeric_cols) > 2:
            suggestions.append({
                'type': 'heatmap',
                'title': 'Correlation Heatmap',
                'description': 'Identify relationships between variables'
            })
        
        logging.info(f"Generated {len(suggestions)} chart suggestions.")
        return suggestions
    
    def export_chart(self, fig: go.Figure, format: str = 'png', width: int = 800, height: int = 600) -> bytes:
        """
        Export chart as image (PNG, SVG) or HTML.
        
        Args:
            fig (go.Figure): The Plotly figure to export.
            format (str): The export format ('png', 'html', 'svg').
            width (int): Width of the exported image.
            height (int): Height of the exported image.

        Returns:
            bytes: The exported chart content as bytes.
        
        Raises:
            ValueError: If an unsupported export format is requested.
        """
        logging.info(f"Exporting chart to format: {format}")
        try:
            if format == 'png':
                return fig.to_image(format='png', width=width, height=height)
            elif format == 'html':
                return fig.to_html().encode('utf-8')
            elif format == 'svg':
                return fig.to_image(format='svg', width=width, height=height)
            else:
                raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            logging.error(f"Error exporting chart to {format} format: {e}", exc_info=True)
            raise
