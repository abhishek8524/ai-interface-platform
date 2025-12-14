export default async (req, res) => {
  const { reqHandler } = await import(
    '../dist/fake-news-detector-ui/server/server.mjs'
  );
  return reqHandler(req, res);
};
