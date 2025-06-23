#!/bin/bash
# =============================================================================
# Git Workflow Setup Script for MLOps Project
# =============================================================================

echo "🚀 Setting up Git Workflow for Demand Stock Forecasting MLOps Project..."

# 1. Configure Git User
echo "📧 Configuring Git user..."
git config --local user.name "Bhupal Lambodhar"
git config --local user.email "btiduwarlambodhar@sandiego.edu"

# 2. Set up branch structure
echo "🌳 Setting up branch structure..."
git checkout -b develop 2>/dev/null || git checkout develop
git checkout -b main 2>/dev/null || git checkout main

# 3. Configure pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-added-large-files
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
        args: [--line-length=100]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --ignore=E203,W503]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile, black]

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-c, pyproject.toml]
        additional_dependencies: ["bandit[toml]"]
EOF

# 4. Install pre-commit
pip install pre-commit
pre-commit install

# 5. Create commit message template
echo "📝 Setting up commit message template..."
cat > .gitmessage << EOF
# Type: Brief description (50 chars max)
#
# Detailed explanation (wrap at 72 chars)
# - What was changed and why
# - Any breaking changes
# - Related issues/tickets
#
# Types: feat, fix, docs, style, refactor, test, chore
# Example: feat: add LSTM model for Chinese produce price forecasting
EOF

git config --local commit.template .gitmessage

# 6. Create pull request template
echo "📋 Setting up PR template..."
mkdir -p .github
cat > .github/pull_request_template.md << EOF
## 🎯 Purpose
Brief description of the changes

## 🔄 Type of Change
- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 🚨 Breaking change
- [ ] 📚 Documentation update
- [ ] 🧹 Code cleanup/refactoring
- [ ] 🧪 Tests

## 🧪 Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Data validation passes

## 📋 Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive data exposed
- [ ] Model performance validated (if applicable)

## 🔗 Related Issues
Closes #issue_number

## 📸 Screenshots (if applicable)
Add screenshots for UI changes
EOF

# 7. Create issue templates
echo "🐛 Setting up issue templates..."
mkdir -p .github/ISSUE_TEMPLATE

cat > .github/ISSUE_TEMPLATE/bug_report.md << EOF
---
name: 🐛 Bug Report
about: Report a bug in the MLOps pipeline
title: '[BUG] '
labels: bug
assignees: btlambodh
---

## 🐛 Bug Description
A clear description of the bug.

## 🔄 Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## 💡 Expected Behavior
What should have happened.

## 📋 Environment
- Branch:
- Python Version:
- AWS Region:
- Model Version:

## 📸 Screenshots/Logs
Add any relevant screenshots or error logs.
EOF

cat > .github/ISSUE_TEMPLATE/feature_request.md << EOF
---
name: ✨ Feature Request
about: Suggest a new feature for the MLOps pipeline
title: '[FEATURE] '
labels: enhancement
assignees: btlambodh
---

## 🎯 Feature Description
A clear description of the proposed feature.

## 💪 Motivation
Why is this feature needed?

## 💡 Proposed Solution
How should this feature work?

## 🔄 Alternatives Considered
What other approaches did you consider?

## 📋 Additional Context
Add any other context about the feature request.
EOF

# 8. Create workflow status badges for README
echo "🏷️ Creating status badges..."
cat > status_badges.md << EOF
# Demand Stock Forecasting MLOps - Project Status

[![CI/CD Pipeline](https://github.com/btlambodh/demand-stock-forecasting-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/btlambodh/demand-stock-forecasting-mlops/actions/workflows/ci-cd.yml)
[![Code Quality](https://github.com/btlambodh/demand-stock-forecasting-mlops/actions/workflows/quality.yml/badge.svg)](https://github.com/btlambodh/demand-stock-forecasting-mlops/actions/workflows/quality.yml)
[![Security Scan](https://github.com/btlambodh/demand-stock-forecasting-mlops/actions/workflows/security.yml/badge.svg)](https://github.com/btlambodh/demand-stock-forecasting-mlops/actions/workflows/security.yml)
[![Model Performance](https://img.shields.io/badge/Model%20MAPE-<15%25-green)](https://github.com/btlambodh/demand-stock-forecasting-mlops)
[![AWS Status](https://img.shields.io/badge/AWS-Active-orange)](https://github.com/btlambodh/demand-stock-forecasting-mlops)

## Quick Commands

\`\`\`bash
# Development workflow
make workflow-dev

# Run tests
make test-full

# Deploy to staging
make workflow-staging

# Start monitoring
make monitoring-start
\`\`\`
EOF

echo "✅ Git workflow setup completed!"
echo ""
echo "🎯 Next Steps:"
echo "1. Push these changes to your repository"
echo "2. Set up branch protection rules in GitHub"
echo "3. Configure repository secrets for AWS"
echo "4. Test the workflow with a small change"
echo ""
echo "📚 Usage:"
echo "  git add ."
echo "  git commit -m 'feat: setup comprehensive MLOps workflow for demand forecasting'"
echo "  git push origin main"
