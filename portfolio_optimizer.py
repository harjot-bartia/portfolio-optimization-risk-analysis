"""
Portfolio Optimization and Risk Analysis
========================================
A comprehensive tool for analyzing stock portfolios, optimizing asset allocation,
and calculating risk metrics.

Features:
- Download historical stock data
- Calculate returns, volatility, and correlation
- Build efficient frontier
- Optimize Sharpe ratio
- Compute risk metrics (VaR, Maximum Drawdown)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PortfolioOptimizer:
    """
    A class for portfolio optimization and risk analysis.
    """
    
    def __init__(self, tickers, start_date=None, end_date=None):
        """
        Initialize the Portfolio Optimizer.
        
        Parameters:
        -----------
        tickers : list
            List of stock ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
        start_date : str, optional
            Start date for historical data (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for historical data (format: 'YYYY-MM-DD')
        """
        self.tickers = tickers
        self.start_date = start_date or (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def download_data(self):
        """Download historical stock data from Yahoo Finance."""
        print(f"Downloading data for {', '.join(self.tickers)}...")
        print(f"Period: {self.start_date} to {self.end_date}\n")
        
        df = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=False)

        # Handle both single-ticker and multi-ticker formats
        if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex: ('Adj Close', 'AAPL') or ('Close','AAPL') etc.
            if 'Adj Close' in df.columns.get_level_values(0):
                self.data = df['Adj Close']
            else:
                self.data = df['Close']
        else:
        # Single index columns: 'Adj Close' or 'Close'
            self.data = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    
        
        
        if len(self.tickers) == 1:
            self.data = self.data.to_frame()
            self.data.columns = self.tickers
        
        print(f"Downloaded {len(self.data)} days of data\n")
        return self.data
    
    def calculate_returns(self):
        """Calculate daily returns from price data."""
        if self.data is None:
            raise ValueError("No data available. Run download_data() first.")
        
        self.returns = self.data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        print("Returns Statistics:")
        print("=" * 60)
        print(f"Mean Daily Returns:\n{self.mean_returns}\n")
        print(f"Annualized Returns:\n{self.mean_returns * 252}\n")
        
        return self.returns
    
    def calculate_volatility(self):
        """Calculate volatility (standard deviation) of returns."""
        if self.returns is None:
            raise ValueError("Returns not calculated. Run calculate_returns() first.")
        
        volatility = self.returns.std()
        annualized_vol = volatility * np.sqrt(252)
        
        print("Volatility Statistics:")
        print("=" * 60)
        print(f"Daily Volatility:\n{volatility}\n")
        print(f"Annualized Volatility:\n{annualized_vol}\n")
        
        return volatility, annualized_vol
    
    def calculate_correlation(self):
        """Calculate correlation matrix between assets."""
        if self.returns is None:
            raise ValueError("Returns not calculated. Run calculate_returns() first.")
        
        corr_matrix = self.returns.corr()
        
        print("Correlation Matrix:")
        print("=" * 60)
        print(corr_matrix)
        print()
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Asset Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return corr_matrix
    
    def portfolio_performance(self, weights):
        """
        Calculate portfolio performance metrics.
        
        Parameters:
        -----------
        weights : array
            Portfolio weights for each asset
            
        Returns:
        --------
        tuple : (returns, volatility, sharpe_ratio)
        """
        returns = np.sum(self.mean_returns * weights) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        sharpe_ratio = returns / volatility
        
        return returns, volatility, sharpe_ratio
    
    def negative_sharpe(self, weights):
        """Negative Sharpe ratio for optimization (scipy minimizes)."""
        return -self.portfolio_performance(weights)[2]
    
    def optimize_sharpe(self):
        """
        Optimize portfolio to maximize Sharpe ratio.
        
        Returns:
        --------
        dict : Optimal weights and performance metrics
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Run calculate_returns() first.")
        
        num_assets = len(self.tickers)
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds: each weight between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(self.negative_sharpe, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        opt_return, opt_vol, opt_sharpe = self.portfolio_performance(optimal_weights)
        
        print("\nOptimal Portfolio (Maximum Sharpe Ratio):")
        print("=" * 60)
        for ticker, weight in zip(self.tickers, optimal_weights):
            print(f"{ticker}: {weight*100:.2f}%")
        
        print(f"\nExpected Annual Return: {opt_return*100:.2f}%")
        print(f"Annual Volatility: {opt_vol*100:.2f}%")
        print(f"Sharpe Ratio: {opt_sharpe:.2f}\n")
        
        return {
            'weights': optimal_weights,
            'return': opt_return,
            'volatility': opt_vol,
            'sharpe_ratio': opt_sharpe
        }
    
    def efficient_frontier(self, num_portfolios=10000):
        """
        Generate efficient frontier with random portfolios.
        
        Parameters:
        -----------
        num_portfolios : int
            Number of random portfolios to generate
            
        Returns:
        --------
        DataFrame : Portfolio characteristics (returns, volatility, Sharpe ratio, weights)
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Run calculate_returns() first.")
        
        print(f"Generating efficient frontier with {num_portfolios} portfolios...")
        
        num_assets = len(self.tickers)
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            # Calculate portfolio metrics
            portfolio_return, portfolio_vol, portfolio_sharpe = self.portfolio_performance(weights)
            
            results[0, i] = portfolio_return
            results[1, i] = portfolio_vol
            results[2, i] = portfolio_sharpe
        
        # Create DataFrame
        portfolios = pd.DataFrame({
            'Returns': results[0],
            'Volatility': results[1],
            'Sharpe_Ratio': results[2]
        })
        
        for i, ticker in enumerate(self.tickers):
            portfolios[ticker] = [w[i] for w in weights_record]
        
        print("Efficient frontier generated!\n")
        
        return portfolios
    
    def plot_efficient_frontier(self, portfolios, optimal_portfolio=None):
        """
        Plot the efficient frontier.
        
        Parameters:
        -----------
        portfolios : DataFrame
            Portfolio characteristics from efficient_frontier()
        optimal_portfolio : dict, optional
            Optimal portfolio from optimize_sharpe()
        """
        plt.figure(figsize=(14, 8))
        
        # Scatter plot of all portfolios
        scatter = plt.scatter(portfolios['Volatility']*100, 
                             portfolios['Returns']*100,
                             c=portfolios['Sharpe_Ratio'],
                             cmap='viridis',
                             alpha=0.5,
                             s=10)
        
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        # Plot optimal portfolio if provided
        if optimal_portfolio:
            plt.scatter(optimal_portfolio['volatility']*100,
                       optimal_portfolio['return']*100,
                       c='red',
                       marker='*',
                       s=500,
                       edgecolors='black',
                       linewidths=2,
                       label='Optimal Portfolio (Max Sharpe)',
                       zorder=5)
        
        plt.xlabel('Volatility (Annual Standard Deviation) %', fontsize=12)
        plt.ylabel('Expected Return %', fontsize=12)
        plt.title('Efficient Frontier', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/efficient_frontier.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Efficient frontier plot saved!\n")
    
    def calculate_var(self, weights, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) using historical simulation.
        
        Parameters:
        -----------
        weights : array
            Portfolio weights
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
        --------
        tuple : (daily_var, annual_var)
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Run calculate_returns() first.")
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Calculate VaR
        var_daily = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        var_annual = var_daily * np.sqrt(252)
        
        print(f"\nValue at Risk (VaR) at {confidence_level*100}% confidence:")
        print("=" * 60)
        print(f"Daily VaR: {var_daily*100:.2f}%")
        print(f"Annual VaR: {var_annual*100:.2f}%")
        print(f"\nInterpretation: There is a {(1-confidence_level)*100}% chance of losing")
        print(f"more than {abs(var_daily)*100:.2f}% in a single day.\n")
        
        return var_daily, var_annual
    
    def calculate_cvar(self, weights, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Parameters:
        -----------
        weights : array
            Portfolio weights
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% CVaR)
            
        Returns:
        --------
        float : CVaR value
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Run calculate_returns() first.")
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Calculate VaR threshold
        var_threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # Calculate CVaR (average of returns below VaR)
        cvar = portfolio_returns[portfolio_returns <= var_threshold].mean()
        
        print(f"Conditional VaR (CVaR) at {confidence_level*100}% confidence:")
        print("=" * 60)
        print(f"CVaR: {cvar*100:.2f}%")
        print(f"\nInterpretation: Given that losses exceed the VaR threshold,")
        print(f"the expected loss is {abs(cvar)*100:.2f}%.\n")
        
        return cvar
    
    def calculate_max_drawdown(self, weights):
        """
        Calculate maximum drawdown for a portfolio.
        
        Parameters:
        -----------
        weights : array
            Portfolio weights
            
        Returns:
        --------
        tuple : (max_drawdown, drawdown_series)
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Run calculate_returns() first.")
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        print("Maximum Drawdown:")
        print("=" * 60)
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"\nInterpretation: The largest peak-to-trough decline in")
        print(f"portfolio value was {abs(max_drawdown)*100:.2f}%.\n")
        
        # Plot drawdown
        plt.figure(figsize=(14, 6))
        drawdown.plot(color='red', linewidth=1.5)
        plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown %', fontsize=12)
        plt.title('Portfolio Drawdown Over Time', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/drawdown_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return max_drawdown, drawdown
    
    def plot_cumulative_returns(self, weights):
        """
        Plot cumulative returns for the portfolio and individual assets.
        
        Parameters:
        -----------
        weights : array
            Portfolio weights
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Run calculate_returns() first.")
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Calculate cumulative returns
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        individual_cumulative = (1 + self.returns).cumprod()
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        # Plot individual assets
        for ticker in self.tickers:
            plt.plot(individual_cumulative.index, 
                    individual_cumulative[ticker],
                    label=ticker,
                    alpha=0.7,
                    linewidth=1.5)
        
        # Plot portfolio
        plt.plot(portfolio_cumulative.index,
                portfolio_cumulative,
                label='Optimized Portfolio',
                color='black',
                linewidth=2.5,
                linestyle='--')
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Returns (Growth of $1)', fontsize=12)
        plt.title('Cumulative Returns: Individual Assets vs Optimized Portfolio', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/cumulative_returns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Cumulative returns plot saved!\n")
    
    def generate_summary_report(self, optimal_portfolio, portfolios):
        """
        Generate a comprehensive summary report.
        
        Parameters:
        -----------
        optimal_portfolio : dict
            Optimal portfolio from optimize_sharpe()
        portfolios : DataFrame
            All portfolios from efficient_frontier()
        """
        report = []
        report.append("=" * 80)
        report.append("PORTFOLIO OPTIMIZATION & RISK ANALYSIS - SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"\nAnalysis Period: {self.start_date} to {self.end_date}")
        report.append(f"Assets Analyzed: {', '.join(self.tickers)}")
        report.append(f"Total Trading Days: {len(self.data)}")
        
        report.append("\n" + "-" * 80)
        report.append("OPTIMAL PORTFOLIO ALLOCATION (Maximum Sharpe Ratio)")
        report.append("-" * 80)
        for ticker, weight in zip(self.tickers, optimal_portfolio['weights']):
            report.append(f"{ticker:10s}: {weight*100:6.2f}%")
        
        report.append("\n" + "-" * 80)
        report.append("PERFORMANCE METRICS")
        report.append("-" * 80)
        report.append(f"Expected Annual Return:    {optimal_portfolio['return']*100:6.2f}%")
        report.append(f"Annual Volatility:         {optimal_portfolio['volatility']*100:6.2f}%")
        report.append(f"Sharpe Ratio:              {optimal_portfolio['sharpe_ratio']:6.2f}")
        
        # Calculate risk metrics
        var_daily, var_annual = self.calculate_var(optimal_portfolio['weights'])
        cvar = self.calculate_cvar(optimal_portfolio['weights'])
        max_dd, _ = self.calculate_max_drawdown(optimal_portfolio['weights'])
        
        report.append("\n" + "-" * 80)
        report.append("RISK METRICS")
        report.append("-" * 80)
        report.append(f"Value at Risk (95%, Daily): {abs(var_daily)*100:6.2f}%")
        report.append(f"Value at Risk (95%, Annual): {abs(var_annual)*100:6.2f}%")
        report.append(f"Conditional VaR (95%):      {abs(cvar)*100:6.2f}%")
        report.append(f"Maximum Drawdown:           {abs(max_dd)*100:6.2f}%")
        
        report.append("\n" + "-" * 80)
        report.append("EFFICIENT FRONTIER STATISTICS")
        report.append("-" * 80)
        report.append(f"Portfolios Generated:       {len(portfolios):,}")
        report.append(f"Return Range:               {portfolios['Returns'].min()*100:.2f}% to {portfolios['Returns'].max()*100:.2f}%")
        report.append(f"Volatility Range:           {portfolios['Volatility'].min()*100:.2f}% to {portfolios['Volatility'].max()*100:.2f}%")
        report.append(f"Sharpe Ratio Range:         {portfolios['Sharpe_Ratio'].min():.2f} to {portfolios['Sharpe_Ratio'].max():.2f}")
        
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save to file
        with open('outputs/portfolio_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\nReport saved to portfolio_report.txt\n")


def main():
    """
    Main function to run portfolio optimization analysis.
    """
    # Example: Analyze a portfolio of tech stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    print("=" * 80)
    print("PORTFOLIO OPTIMIZATION AND RISK ANALYSIS")
    print("=" * 80)
    print()
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(tickers)
    
    # Download data
    optimizer.download_data()
    
    # Calculate returns and statistics
    optimizer.calculate_returns()
    optimizer.calculate_volatility()
    optimizer.calculate_correlation()
    
    # Optimize portfolio
    optimal_portfolio = optimizer.optimize_sharpe()
    
    # Generate efficient frontier
    portfolios = optimizer.efficient_frontier(num_portfolios=10000)
    
    # Plot efficient frontier
    optimizer.plot_efficient_frontier(portfolios, optimal_portfolio)
    
    # Calculate risk metrics
    optimizer.calculate_var(optimal_portfolio['weights'])
    optimizer.calculate_cvar(optimal_portfolio['weights'])
    optimizer.calculate_max_drawdown(optimal_portfolio['weights'])
    
    # Plot cumulative returns
    optimizer.plot_cumulative_returns(optimal_portfolio['weights'])
    
    # Generate summary report
    optimizer.generate_summary_report(optimal_portfolio, portfolios)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • correlation_matrix.png")
    print("  • efficient_frontier.png")
    print("  • drawdown_chart.png")
    print("  • cumulative_returns.png")
    print("  • portfolio_report.txt")
    print()


if __name__ == "__main__":
    main()
