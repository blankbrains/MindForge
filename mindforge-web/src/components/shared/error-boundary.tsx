import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  retryKey: number;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null, retryKey: 0 };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, retryKey: 0 };
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, retryKey: this.state.retryKey + 1 });
  };

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-center dark:border-red-800 dark:bg-red-950">
              <h3 className="text-lg font-medium text-red-800 dark:text-red-200">
                发生错误
              </h3>
              <p className="mt-2 text-sm text-red-600 dark:text-red-300">
                {this.state.error?.message ?? "未知错误"}
              </p>
              <button
                type="button"
                onClick={this.handleRetry}
                className="mt-4 rounded-md bg-red-100 px-4 py-2 text-sm font-medium text-red-700 hover:bg-red-200 dark:bg-red-900 dark:text-red-200 dark:hover:bg-red-800"
              >
                重试
              </button>
            </div>
          </div>
        )
      );
    }
    // key 自增强制重挂子树，避免重试后白屏
    return <div key={this.state.retryKey}>{this.props.children}</div>;
  }
}
