export default {
  extends: ["@commitlint/config-conventional"],
  // CarryCtx's prepare-commit-msg hook prefixes subjects with the owning task,
  // for example "[AI-0004] docs: ..."; accept and ignore that prefix.
  parserPreset: {
    parserOpts: {
      headerPattern: /^(?:\[AI-\d+\]\s+)?(\w*)(?:\((.*)\))?!?: (.*)$/,
      headerCorrespondence: ["type", "scope", "subject"],
    },
  },
};
