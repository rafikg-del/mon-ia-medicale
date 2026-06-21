import { useCallback, useEffect, useState } from "react";
import { getAllPosts, savePost, deletePost, clearAllPosts, updatePost } from "../utils/db.js";

/**
 * Source de vérité pour l'historique des posts capturés.
 * Gère le chargement, l'ajout, la suppression et la mise à jour (tags).
 */
export function useScraper() {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    const all = await getAllPosts();
    setPosts(all);
    return all;
  }, []);

  useEffect(() => {
    refresh().finally(() => setLoading(false));
  }, [refresh]);

  // Ajoute une capture ; renvoie l'enregistrement et s'il s'agit d'un doublon.
  const capture = useCallback(
    async (data) => {
      const before = await getAllPosts();
      const wasKnown = before.some(
        (p) => p.url === data.url && p.content === data.content,
      );
      const record = await savePost(data);
      await refresh();
      return { record, duplicate: wasKnown };
    },
    [refresh],
  );

  const remove = useCallback(
    async (id) => {
      await deletePost(id);
      await refresh();
    },
    [refresh],
  );

  const clearAll = useCallback(async () => {
    await clearAllPosts();
    await refresh();
  }, [refresh]);

  const setTags = useCallback(
    async (id, tags) => {
      await updatePost(id, { tags });
      await refresh();
    },
    [refresh],
  );

  return { posts, loading, capture, remove, clearAll, setTags, refresh };
}
