"""
Example: Quick Portfolio Analysis
==================================
This script demonstrates a quick portfolio analysis workflow.
"""

from portfolio_optimizer import PortfolioOptimizer

def quick_analysis(tickers, name="Portfolio"):
    """
    Perform a quick portfolio analysis.
    
    Parameters:
    -----------
    tickers : list
        List of stock tickers
    name : str
        Name for the portfolio
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {name}")
    print(f"{'='*80}\n")
    
    # Initialize
    optimizer = PortfolioOptimizer(tickers)
    
    # Download data
    optimizer.download_data()
    
    # Calculate statistics
    optimizer.calculate_returns()
    optimizer.calculate_volatility()
    optimizer.calculate_correlation()
    
    # Optimize
    optimal = optimizer.optimize_sharpe()
    
    # Generate efficient frontier
    portfolios = optimizer.efficient_frontier(num_portfolios=5000)
    
    # Visualizations
    optimizer.plot_efficient_frontier(portfolios, optimal)
    optimizer.plot_cumulative_returns(optimal['weights'])
    
    # Risk metrics
    optimizer.calculate_var(optimal['weights'])
    optimizer.calculate_cvar(optimal['weights'])
    optimizer.calculate_max_drawdown(optimal['weights'])
    
    # Generate report
    optimizer.generate_summary_report(optimal, portfolios)
    
    return optimal, portfolios


if __name__ == "__main__":
    # Example 1: Tech Portfolio
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
    tech_optimal, tech_portfolios = quick_analysis(tech_stocks, "Tech Portfolio")
    
    # Example 2: Diversified Portfolio
    print("\n\n")
    diversified_stocks = ['SPY', 'QQQ', 'GLD', 'TLT', 'VNQ']
    div_optimal, div_portfolios = quick_analysis(diversified_stocks, "Diversified ETF Portfolio")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\nTech Portfolio:")
    print(f"  Return: {tech_optimal['return']*100:.2f}%")
    print(f"  Volatility: {tech_optimal['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {tech_optimal['sharpe_ratio']:.2f}")
    
    print(f"\nDiversified Portfolio:")
    print(f"  Return: {div_optimal['return']*100:.2f}%")
    print(f"  Volatility: {div_optimal['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {div_optimal['sharpe_ratio']:.2f}")
    print()