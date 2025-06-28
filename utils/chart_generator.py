import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

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
            'gridcolor': '#E9ECEF'
        }
    
    def create_chart(self, data: pd.DataFrame, chart_config: Dict[str, Any]) -> go.Figure:
        """
        Create a chart based on configuration.
        
        Args:
            data: DataFrame containing the data
            chart_config: Configuration dictionary specifying chart type and parameters
            
        Returns:
            Plotly figure object
        """
        try:
            chart_type = chart_config.get('type', 'bar')
            
            if chart_type == 'bar':
                return self.create_bar_chart(data, chart_config)
            elif chart_type == 'line':
                return self.create_line_chart(data, chart_config)
            elif chart_type == 'scatter':
                return self.create_scatter_plot(data, chart_config)
            elif chart_type == 'pie':
                return self.create_pie_chart(data, chart_config)
            elif chart_type == 'histogram':
                return self.create_histogram(data, chart_config)
            elif chart_type == 'box':
                return self.create_box_plot(data, chart_config)
            elif chart_type == 'heatmap':
                return self.create_heatmap(data, chart_config)
            elif chart_type == 'area':
                return self.create_area_chart(data, chart_config)
            elif chart_type == 'violin':
                return self.create_violin_plot(data, chart_config)
            elif chart_type == 'sunburst':
                return self.create_sunburst_chart(data, chart_config)
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
                
        except Exception as e:
            return self._create_error_chart(f"Error creating chart: {str(e)}")
    
    def create_bar_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a bar chart"""
        
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
        
        self._apply_theme(fig)
        return fig
    
    def create_line_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a line chart"""
        
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        title = config.get('title', 'Line Chart')
        
        fig = px.line(data, x=x_col, y=y_col, color=color_col,
                     title=title, color_discrete_sequence=self.color_palette)
        
        # Add markers
        fig.update_traces(mode='lines+markers')
        
        self._apply_theme(fig)
        return fig
    
    def create_scatter_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a scatter plot"""
        
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        size_col = config.get('size_column')
        title = config.get('title', 'Scatter Plot')
        
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col, size=size_col,
                        title=title, color_discrete_sequence=self.color_palette)
        
        # Add trendline if requested
        if config.get('trendline'):
            fig.add_scatter(x=data[x_col], y=data[y_col], mode='lines', 
                          name='Trendline', line=dict(color='red', dash='dash'))
        
        self._apply_theme(fig)
        return fig
    
    def create_pie_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a pie chart"""
        
        values_col = config.get('values_column')
        names_col = config.get('names_column')
        title = config.get('title', 'Pie Chart')
        
        # If values column is not specified, use value counts
        if not values_col and names_col:
            pie_data = data[names_col].value_counts().reset_index()
            pie_data.columns = ['names', 'values']
            fig = px.pie(pie_data, values='values', names='names', title=title,
                        color_discrete_sequence=self.color_palette)
        else:
            fig = px.pie(data, values=values_col, names=names_col, title=title,
                        color_discrete_sequence=self.color_palette)
        
        self._apply_theme(fig)
        return fig
    
    def create_histogram(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a histogram"""
        
        x_col = config.get('x_column')
        color_col = config.get('color_column')
        title = config.get('title', 'Histogram')
        bins = config.get('bins', 30)
        
        fig = px.histogram(data, x=x_col, color=color_col, title=title,
                          nbins=bins, color_discrete_sequence=self.color_palette)
        
        self._apply_theme(fig)
        return fig
    
    def create_box_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a box plot"""
        
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        title = config.get('title', 'Box Plot')
        
        fig = px.box(data, x=x_col, y=y_col, color=color_col, title=title,
                    color_discrete_sequence=self.color_palette)
        
        self._apply_theme(fig)
        return fig
    
    def create_heatmap(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a heatmap"""
        
        title = config.get('title', 'Heatmap')
        
        # Create correlation matrix for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return self._create_error_chart("Heatmap requires at least 2 numeric columns")
        
        corr_matrix = data[numeric_cols].corrwith(data[numeric_cols], method='pearson')
        fig = px.imshow(corr_matrix, title=title, color_continuous_scale='RdBu_r',
                       aspect='auto')
        
        # Add correlation values as text
        fig.update_traces(text=np.around(np.array(corr_matrix.values), decimals=2),
                         texttemplate="%{text}")
        
        self._apply_theme(fig)
        return fig
    
    def create_area_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create an area chart"""
        
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        title = config.get('title', 'Area Chart')
        
        fig = px.area(data, x=x_col, y=y_col, color=color_col, title=title,
                     color_discrete_sequence=self.color_palette)
        
        self._apply_theme(fig)
        return fig
    
    def create_violin_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a violin plot"""
        
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        title = config.get('title', 'Violin Plot')
        
        fig = px.violin(data, x=x_col, y=y_col, color=color_col, title=title,
                       color_discrete_sequence=self.color_palette)
        
        self._apply_theme(fig)
        return fig
    
    def create_sunburst_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a sunburst chart"""
        
        path_cols = config.get('path_columns', [])
        values_col = config.get('values_column')
        title = config.get('title', 'Sunburst Chart')
        
        if not path_cols:
            return self._create_error_chart("Sunburst chart requires path columns")
        
        fig = px.sunburst(data, path=path_cols, values=values_col, title=title,
                         color_discrete_sequence=self.color_palette)
        
        self._apply_theme(fig)
        return fig
    
    def create_multi_chart(self, data: pd.DataFrame, configs: List[Dict[str, Any]]) -> go.Figure:
        """Create a multi-chart layout"""
        
        if not configs:
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
            subplot_titles=[config.get('title', f'Chart {i+1}') for i, config in enumerate(configs)]
        )
        
        # Add charts to subplots
        for i, config in enumerate(configs):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Create individual chart
            chart = self.create_chart(data, config)
            
            # Add traces to subplot
            for trace in chart.data:
                fig.add_trace(trace, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title_text="Multi-Chart Dashboard",
            showlegend=True,
            plot_bgcolor=self.layout_theme['plot_bgcolor'],
            paper_bgcolor=self.layout_theme['paper_bgcolor'],
            font_color=self.layout_theme['font_color'],
            gridcolor=self.layout_theme['gridcolor']
        )
        
        return fig
    
    def create_dashboard(self, data: pd.DataFrame, dashboard_config: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive dashboard"""
        
        # Auto-generate dashboard based on data characteristics
        charts = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Add correlation heatmap if multiple numeric columns
        if len(numeric_cols) > 1:
            charts.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix'
            })
        
        # Add distribution charts for numeric columns
        for col in numeric_cols[:2]:  # Limit to first 2
            charts.append({
                'type': 'histogram',
                'x_column': col,
                'title': f'Distribution of {col}'
            })
        
        # Add bar charts for categorical vs numeric
        if categorical_cols and numeric_cols:
            charts.append({
                'type': 'bar',
                'x_column': categorical_cols[0],
                'y_column': numeric_cols[0],
                'title': f'{numeric_cols[0]} by {categorical_cols[0]}'
            })
        
        # Create multi-chart layout
        return self.create_multi_chart(data, charts)
    
    def _apply_theme(self, fig: go.Figure):
        """Apply consistent theme to charts"""
        
        fig.update_layout(
            plot_bgcolor=self.layout_theme['plot_bgcolor'],
            paper_bgcolor=self.layout_theme['paper_bgcolor'],
            font=dict(color=self.layout_theme['font_color']),
            title=dict(
                text=fig.layout["title"]["text"] if "title" in fig.layout and "text" in fig.layout["title"] else '',
                font=dict(size=20, color=self.layout_theme['font_color'])
            ),
            legend=dict(
                title=dict(text='Legend', font=dict(size=14, color=self.layout_theme['font_color'])),
                font=dict(size=12, color=self.layout_theme['font_color'])
            ),
            height=600,
            width=800,
            title_font_size=16,
            title_font_color=self.layout_theme['font_color'],
            xaxis=dict(gridcolor=self.layout_theme['gridcolor']),
            yaxis=dict(gridcolor=self.layout_theme['gridcolor'])
        )
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create an error chart when visualization fails"""
        
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title="Chart Generation Error",
            plot_bgcolor=self.layout_theme['plot_bgcolor'],
            paper_bgcolor=self.layout_theme['paper_bgcolor'],
            font_color=self.layout_theme['font_color'],
            gridcolor=self.layout_theme['gridcolor'],
            height=400,
            width=600
    
        )
        
        return fig
    
    def get_chart_suggestions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get suggested chart configurations based on data characteristics"""
        
        suggestions = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Bar chart suggestions
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
        
        return suggestions
    
    def export_chart(self, fig: go.Figure, format: str = 'png', width: int = 800, height: int = 600) -> bytes:
        """Export chart as image or HTML"""
        
        if format == 'png':
            return fig.to_image(format='png', width=width, height=height)
        elif format == 'html':
            return fig.to_html().encode('utf-8')
        elif format == 'svg':
            return fig.to_image(format='svg', width=width, height=height)
        else:
            raise ValueError(f"Unsupported export format: {format}")
