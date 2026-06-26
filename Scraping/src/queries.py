"""GraphQL queries used by the AniList scraper."""

ANIME_PAGE_QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      currentPage
      hasNextPage
    }
    media(type: ANIME, sort: POPULARITY_DESC) {
      id
      idMal
      title {
        romaji
        english
        native
      }
      type
      format
      source
      episodes
      duration
      status
      season
      seasonYear
      startDate {
        year
        month
        day
      }
      endDate {
        year
        month
        day
      }
      averageScore
      meanScore
      popularity
      favourites
      genres
      tags {
        name
        rank
      }
      studios {
        nodes {
          name
        }
      }
      description(asHtml: false)
      coverImage {
        large
      }
      isAdult
    }
  }
}
"""

ANIME_PAGE_BY_POPULARITY_QUERY = """
query ($page: Int, $perPage: Int, $popularityLesser: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      currentPage
      hasNextPage
    }
    media(type: ANIME, sort: POPULARITY_DESC, popularity_lesser: $popularityLesser) {
      id
      idMal
      title {
        romaji
        english
        native
      }
      type
      format
      source
      episodes
      duration
      status
      season
      seasonYear
      startDate {
        year
        month
        day
      }
      endDate {
        year
        month
        day
      }
      averageScore
      meanScore
      popularity
      favourites
      genres
      tags {
        name
        rank
      }
      studios {
        nodes {
          name
        }
      }
      description(asHtml: false)
      coverImage {
        large
      }
      isAdult
    }
  }
}
"""
