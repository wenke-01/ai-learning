# MCP Server Setup Guide

## What is an MCP Server?

**MCP (Model Context Protocol)** is a protocol that allows AI assistants like Claude to connect to external services and tools. Think of MCP servers as "plugins" or "extensions" that give Claude new capabilities.

### Key Concepts

- **MCP Server**: A service that provides specific functionality (like GitHub integration, database access, etc.)
- **Transport Types**: How Claude communicates with the server (HTTP, SSE, or stdio)
- **Scope**: Where the configuration is stored (local, user, or project)

### Benefits of MCP Servers

1. **Extended Functionality**: Access to external APIs and services
2. **Real-time Data**: Connect to live databases, APIs, and services
3. **Automation**: Automate tasks across multiple platforms
4. **Customization**: Add custom tools specific to your workflow

---

## Common MCP Servers

| Server | Purpose | Transport Type |
|--------|---------|----------------|
| GitHub | Manage repositories, issues, PRs | HTTP |
| Sentry | Monitor errors and performance | HTTP |
| Asana | Project management | SSE |
| PostgreSQL | Database queries | stdio |
| Slack | Team communication | HTTP |

---

## How to Configure an MCP Server

### Basic Syntax

```bash
claude mcp add [options] <name> <commandOrUrl> [args...]
```

### Common Options

- `-t, --transport <type>`: Specify transport type (http, sse, stdio)
- `-H, --header <header>`: Add HTTP headers (for authentication)
- `-e, --env <env>`: Set environment variables
- `-s, --scope <scope>`: Set configuration scope (local, user, project)

---

## Example: Setting Up GitHub MCP Server

This is a step-by-step guide based on our actual setup process.

### Step 1: Create GitHub Personal Access Token

1. Go to GitHub Settings: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Configure the token:
   - **Note**: `Claude Code MCP`
   - **Expiration**: Choose your preference
   - **Scopes**: Select these permissions:
     - ✅ `repo` (Full control of repositories)
     - ✅ `read:org` (Read organization data)
     - ✅ `user` (Read user data)
     - ✅ `workflow` (Update GitHub Actions)
4. Click **"Generate token"**
5. **COPY THE TOKEN** immediately (it looks like `ghp_xxxxxxxxxxxx`)

### Step 2: Add the MCP Server

```bash
claude mcp add github https://api.githubcopilot.com/mcp/ \
  --transport http \
  -H "Authorization: Bearer YOUR_GITHUB_TOKEN_HERE"
```

**Replace `YOUR_GITHUB_TOKEN_HERE` with your actual token!**

### Step 3: Verify the Connection

```bash
claude mcp list
```

You should see:
```
✓ github: https://api.githubcopilot.com/mcp/ (HTTP) - Connected
```

---

## Managing MCP Servers

### List All Configured Servers

```bash
claude mcp list
```

### View Server Details

```bash
claude mcp get <server-name>
```

Example:
```bash
claude mcp get github
```

### Remove a Server

```bash
claude mcp remove <server-name> -s <scope>
```

Example:
```bash
claude mcp remove github -s local
```

### Update a Server

To update an MCP server configuration:
1. Remove the existing server
2. Add it again with new configuration

```bash
claude mcp remove github -s local
claude mcp add github https://api.githubcopilot.com/mcp/ \
  --transport http \
  -H "Authorization: Bearer NEW_TOKEN"
```

---

## Configuration Scopes

### Local Scope (Default)
- Stored in: `.claude.json` in your project directory
- **Private to you** in this specific project
- Useful for project-specific configurations

### User Scope
- Stored in: Your user configuration file
- Available to you across all projects
- Useful for personal tools you use everywhere

### Project Scope
- Stored in: Project's `.claude.json` (shared)
- Available to all team members
- Useful for team-wide integrations

**To specify scope:**
```bash
claude mcp add -s user github https://api.githubcopilot.com/mcp/ ...
```

---

## Transport Types Explained

### HTTP Transport
- Used for REST APIs
- Requires URL endpoint
- Example: GitHub, Sentry

```bash
claude mcp add --transport http github https://api.example.com/mcp/
```

### SSE (Server-Sent Events)
- Used for real-time streaming data
- Maintains persistent connection
- Example: Asana

```bash
claude mcp add --transport sse asana https://mcp.asana.com/sse
```

### stdio (Standard Input/Output)
- Used for command-line tools
- Runs local processes
- Example: Database connections

```bash
claude mcp add --transport stdio postgres --env DB_URL=postgresql://localhost -- npx postgres-mcp
```

---

## Security Best Practices

### 1. Token Management
- ✅ Use tokens with minimum required permissions
- ✅ Set expiration dates on tokens
- ✅ Rotate tokens regularly
- ❌ Never commit tokens to version control

### 2. Configuration Files
- ✅ Add `.claude.json` to `.gitignore` if it contains secrets
- ✅ Use environment variables for sensitive data
- ✅ Use project scope only for non-sensitive configurations

### 3. Token Revocation
If you accidentally expose a token:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Find the token and click **"Revoke"**
3. Generate a new token
4. Update your MCP configuration

---

## Troubleshooting

### Server Won't Connect

**Check 1: Verify token is valid**
```bash
# For GitHub, test with curl:
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.github.com/user
```

**Check 2: Check server configuration**
```bash
claude mcp get <server-name>
```

**Check 3: Remove and re-add the server**
```bash
claude mcp remove <server-name> -s local
claude mcp add <server-name> ...
```

### Authentication Errors

- Verify your token has the correct scopes/permissions
- Check if the token has expired
- Ensure the Authorization header is formatted correctly

### Command Not Found

Make sure Claude Code CLI is properly installed:
```bash
claude --version
```

---

## Real-World Use Cases

### 1. Automated GitHub Workflows
- Create issues and PRs directly from Claude
- Review code and add comments
- Manage repository settings

### 2. Database Operations
- Query databases for information
- Update records based on AI analysis
- Generate reports from data

### 3. Multi-Tool Integration
- Combine GitHub + Slack + Sentry
- Automated incident response
- Cross-platform workflow automation

---

## Getting Help

### View Help for MCP Commands

```bash
claude mcp --help
claude mcp add --help
```

### Official Documentation

- Claude Code Documentation: Check with `claude --help`
- MCP Protocol: https://modelcontextprotocol.io/

---

## Quick Reference Card

```bash
# List all servers
claude mcp list

# Add HTTP server
claude mcp add <name> <url> --transport http -H "Header: Value"

# Add stdio server
claude mcp add <name> --transport stdio -e KEY=value -- command args

# Get server info
claude mcp get <name>

# Remove server
claude mcp remove <name> -s local

# Test connection
claude mcp list  # Shows ✓ or ✗ for each server
```

---

## Conclusion

MCP servers transform Claude Code from a simple AI assistant into a powerful automation platform. By connecting to services like GitHub, databases, and APIs, you can build sophisticated workflows that combine AI intelligence with real-world data and actions.

Start with one server (like GitHub), get comfortable with it, then explore adding more based on your needs!

---

**Created**: December 10, 2025
**Author**: AI Learning Journey
**Version**: 1.0