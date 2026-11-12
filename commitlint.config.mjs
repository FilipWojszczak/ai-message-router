export default {
    extends: ['@commitlint/config-conventional'],
    rules: {
        // Rule: 2 = error, 'always' = always check, and the list of allowed scopes
        'scope-enum': [
            2,
            'always',
            [
                'api',          // FastAPI
                'agent',        // AI Agent
                'email',        // Email feature
                'docker',       // Dockerfile / docker-compose
                'ci',           // GitHub Actions
                'docs',         // Documentation
                'deps'          // Dependency changes (uv, pyproject.toml)
            ]
        ],
        // small letters only for scope
        'scope-case': [2, 'always', 'lower-case']
    }
};
